# -*- coding: utf-8 -*-
""" Run: %run /content/Sentiment-Analysis/sentiment_analysis_phobert.py """

import importlib.util
import subprocess
import sys

packages = {
    "transformers": "transformers",
    "sentencepiece": "sentencepiece",
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "pyvi": "pyvi",
    "accelerate": "accelerate",
    "datasets": "datasets",
    "tqdm": "tqdm",
}

missing = [
    package
    for module, package in packages.items()
    if importlib.util.find_spec(module) is None
]

if missing:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", *missing
    ])
from pathlib import Path
import json
import re
import inspect

import numpy as np
import pandas as pd
import torch
from pyvi import ViTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_PATH = Path(__file__).resolve().parent
DATA_PATH = PROJECT_PATH / "data" / "processed"
MODEL_NAME = "vinai/phobert-base"
MODEL_OUTPUT_PATH = PROJECT_PATH / "phobert_model"
TRAIN_OUTPUT_PATH = PROJECT_PATH / "outputs"

MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
SEED = 42
MIN_POSITIVES_FOR_TUNING = 5
THRESHOLD_CANDIDATES = np.arange(0.10, 0.901, 0.025)


# ---------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------

def extract_labels(label_text):
    """Extract labels enclosed in braces from a CSV label cell."""
    if pd.isna(label_text):
        return []
    return re.findall(r"\{([^{}]+)\}", str(label_text))


def load_data():
    """Load train/dev/test CSVs and validate their expected columns."""
    paths = {
        "train": DATA_PATH / "Train.csv",
        "dev": DATA_PATH / "Dev.csv",
        "test": DATA_PATH / "Test.csv",
    }
    data = {}

    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file {path}. "
                "Hãy kiểm tra cấu trúc data/processed trong repository."
            )

        df = pd.read_csv(path)
        missing_columns = {"comment", "label"} - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"{path.name} thiếu cột bắt buộc: {sorted(missing_columns)}"
            )
        if df["label"].isna().any():
            raise ValueError(f"{path.name} có nhãn bị rỗng.")
        data[split] = df

    return data["train"], data["dev"], data["test"]


def prepare_labels(train_df, dev_df, test_df):
    """Fit label binarizer on Train only and validate Dev/Test labels."""
    y_train_list = train_df["label"].apply(extract_labels).tolist()
    y_dev_list = dev_df["label"].apply(extract_labels).tolist()
    y_test_list = test_df["label"].apply(extract_labels).tolist()

    if any(not labels for labels in y_train_list):
        raise ValueError("Train có bình luận không có nhãn hợp lệ.")

    train_classes = {label for labels in y_train_list for label in labels}
    dev_classes = {label for labels in y_dev_list for label in labels}
    test_classes = {label for labels in y_test_list for label in labels}

    unseen_dev = sorted(dev_classes - train_classes)
    unseen_test = sorted(test_classes - train_classes)
    if unseen_dev or unseen_test:
        raise ValueError(
            "Có nhãn trong Dev/Test chưa xuất hiện trong Train. "
            f"Dev: {unseen_dev}; Test: {unseen_test}. "
            "Cần kiểm tra lại cách chia dữ liệu trước khi huấn luyện."
        )

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train_list)
    y_dev = mlb.transform(y_dev_list)
    y_test = mlb.transform(y_test_list)

    print("===== THỐNG KÊ DỮ LIỆU =====")
    print(f"Train: {len(train_df):,} mẫu")
    print(f"Dev:   {len(dev_df):,} mẫu")
    print(f"Test:  {len(test_df):,} mẫu")
    print(f"Số nhãn: {len(mlb.classes_)}")
    print("Nhãn:", list(mlb.classes_))
    print("Comment rỗng - Train:", train_df["comment"].isna().sum())

    return mlb, y_train, y_dev, y_test


# ---------------------------------------------------------------------
# Dataset and preprocessing
# ---------------------------------------------------------------------

class PhoBERTDataset(Dataset):
    """Tokenize each example on demand to avoid storing all tensors in RAM."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = ["" if pd.isna(text) else str(text) for text in texts]
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = ViTokenizer.tokenize(self.texts[idx])
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = self.labels[idx]
        return item


# ---------------------------------------------------------------------
# Metrics and threshold optimization
# ---------------------------------------------------------------------

def sigmoid_np(logits):
    """Numerically stable enough sigmoid for model logits."""
    logits = np.clip(np.asarray(logits), -50, 50)
    return 1.0 / (1.0 + np.exp(-logits))


def compute_metrics(eval_pred):
    """Trainer metrics at the default 0.5 threshold."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = (sigmoid_np(logits) >= 0.5).astype(int)
    labels = np.asarray(labels).astype(int)

    return {
        "precision_micro": precision_score(
            labels, predictions, average="micro", zero_division=0
        ),
        "recall_micro": recall_score(
            labels, predictions, average="micro", zero_division=0
        ),
        "f1_micro": f1_score(
            labels, predictions, average="micro", zero_division=0
        ),
        "f1_macro": f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "subset_accuracy": accuracy_score(labels, predictions),
    }


def optimize_thresholds(dev_probs, y_dev, classes):
    """Choose a per-label threshold using Dev only."""
    thresholds = np.full(y_dev.shape[1], 0.5, dtype=float)

    for j, label in enumerate(classes):
        positives = int(y_dev[:, j].sum())
        if positives < MIN_POSITIVES_FOR_TUNING:
            continue

        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in THRESHOLD_CANDIDATES:
            predictions = (dev_probs[:, j] >= threshold).astype(int)
            score = f1_score(y_dev[:, j], predictions, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)

        thresholds[j] = best_threshold

    return thresholds


def print_metrics(title, y_true, y_pred):
    print(f"\n===== {title} =====")
    print(
        "Precision Micro:",
        f"{precision_score(y_true, y_pred, average='micro', zero_division=0):.4f}",
    )
    print(
        "Recall Micro:   ",
        f"{recall_score(y_true, y_pred, average='micro', zero_division=0):.4f}",
    )
    print(
        "F1 Micro:       ",
        f"{f1_score(y_true, y_pred, average='micro', zero_division=0):.4f}",
    )
    print(
        "F1 Macro:       ",
        f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}",
    )
    print(f"Subset Accuracy: {accuracy_score(y_true, y_pred):.4f}")


# ---------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------

def main():
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Chưa bật GPU; huấn luyện có thể rất chậm.")

    train_df, dev_df, test_df = load_data()
    mlb, y_train, y_dev, y_test = prepare_labels(train_df, dev_df, test_df)

    label2id = {label: i for i, label in enumerate(mlb.classes_)}
    id2label = {i: label for label, i in label2id.items()}

    print("\nĐang tải tokenizer và PhoBERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(mlb.classes_),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    train_dataset = PhoBERTDataset(
        train_df["comment"].tolist(), y_train, tokenizer, MAX_LENGTH
    )
    dev_dataset = PhoBERTDataset(
        dev_df["comment"].tolist(), y_dev, tokenizer, MAX_LENGTH
    )
    test_dataset = PhoBERTDataset(
        test_df["comment"].tolist(), y_test, tokenizer, MAX_LENGTH
    )

    # Transformers changed evaluation_strategy to eval_strategy in newer versions.
    training_args_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = (
        "eval_strategy"
        if "eval_strategy" in training_args_params
        else "evaluation_strategy"
    )

    training_config = {
        "output_dir": str(TRAIN_OUTPUT_PATH),
        "num_train_epochs": EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_steps": WARMUP_STEPS,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "logging_steps": 50,
        "save_total_limit": 2,
        "fp16": torch.cuda.is_available(),
        "report_to": "none",
        "seed": SEED,
        "data_seed": SEED,
    }
    training_config[eval_key] = "epoch"
    training_args = TrainingArguments(**training_config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\n===== BẮT ĐẦU HUẤN LUYỆN =====")
    trainer.train()

    MODEL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_OUTPUT_PATH))
    tokenizer.save_pretrained(str(MODEL_OUTPUT_PATH))
    print("Đã lưu mô hình tại:", MODEL_OUTPUT_PATH)

    # Select thresholds using Dev only. Test is used for final evaluation.
    dev_output = trainer.predict(dev_dataset)
    test_output = trainer.predict(test_dataset)

    dev_probs = sigmoid_np(
        dev_output.predictions[0]
        if isinstance(dev_output.predictions, tuple)
        else dev_output.predictions
    )
    test_logits = (
        test_output.predictions[0]
        if isinstance(test_output.predictions, tuple)
        else test_output.predictions
    )
    test_probs = sigmoid_np(test_logits)

    y_dev_true = np.asarray(y_dev).astype(int)
    y_test_true = np.asarray(y_test).astype(int)

    thresholds = optimize_thresholds(dev_probs, y_dev_true, mlb.classes_)
    baseline_predictions = (test_probs >= 0.5).astype(int)
    tuned_predictions = (test_probs >= thresholds.reshape(1, -1)).astype(int)

    print_metrics("TEST - ngưỡng mặc định 0.5", y_test_true, baseline_predictions)
    print_metrics("TEST - ngưỡng tối ưu trên Dev", y_test_true, tuned_predictions)

    print("\n===== BÁO CÁO THEO TỪNG NHÃN =====")
    print(
        classification_report(
            y_test_true,
            tuned_predictions,
            target_names=mlb.classes_,
            zero_division=0,
            digits=4,
        )
    )

    thresholds_path = MODEL_OUTPUT_PATH / "thresholds.json"
    with thresholds_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                label: float(threshold)
                for label, threshold in zip(mlb.classes_, thresholds)
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print("Đã lưu ngưỡng tại:", thresholds_path)


if __name__ == "__main__":
    main()
