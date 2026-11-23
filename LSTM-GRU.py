from collections import Counter
from pathlib import Path
import re
import numpy as np
import pandas as pd
from pyvi import ViTokenizer
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

# 1. Đọc dữ liệu và mã hóa ba cảm xúc ở cấp bình luận
SENTIMENTS = ("Positive", "Negative", "Neutral")
MAX_WORDS = 10000


def read_split(path):
    frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    texts, targets = [], []
    skipped = 0
    for row in frame.itertuples(index=False):
        # Một bình luận có thể vừa khen pin vừa chê camera.
        # Các khía cạnh có cùng cảm xúc chỉ bật một vị trí trong vector đích.
        tags = re.findall(r"\{([^{}]+)\}", row.label)
        if not tags or not re.fullmatch(r"(?:\{[^{}]+\};?)+", row.label):
            raise ValueError(f"{path.name}, index={row.index}: nhãn sai định dạng")
        sentiments = set()
        for tag in tags:
            if tag == "OTHERS":
                continue
            aspect, sep, sentiment = tag.rpartition("#")
            if not aspect or not sep or sentiment not in SENTIMENTS:
                raise ValueError(f"{path.name}, index={row.index}: nhãn lạ {tag!r}")
            sentiments.add(sentiment)
        if not sentiments:
            # OTHERS không cho biết cảm xúc. Ghi nhận số dòng thiếu nhãn đích.
            skipped += 1
            continue
        if not row.comment.strip():
            raise ValueError(f"{path.name}, index={row.index}: thiếu bình luận")
        texts.append(row.comment)
        targets.append([int(name in sentiments) for name in SENTIMENTS])

    print(f"{path.name}: dùng {len(texts)}/{len(frame)} bình luận, "
          f"bỏ qua {skipped} dòng chỉ có OTHERS.")
    return texts, np.asarray(targets, dtype=np.float32)


def tokenize_vi(text):
    return ViTokenizer.tokenize(text).lower().split()


# 2. Chuyển chuỗi từ thành PyTorch Tensor và DataLoader
def make_loader(texts, targets, vocab, shuffle=False):
    # Giữ toàn bộ từ của mỗi bình luận. ID 0 là padding, ID 1 là từ ngoài từ điển.
    sequences = [torch.tensor([vocab.get(word, 1) for word in words] or [1],
                              dtype=torch.long) for words in texts]
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    dataset = TensorDataset(padded, lengths, torch.from_numpy(targets))
    return DataLoader(dataset, batch_size=64, shuffle=shuffle)


# 3. Xây dựng mô hình LSTM + GRU kết hợp

class HybridLSTMGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=len(SENTIMENTS), dropout=0.3):
        super(HybridLSTMGRU, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)

        # LSTM layer (trả về toàn bộ sequence)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)

        # GRU layer (nhận đầu vào là chuỗi từ LSTM)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        # x: (batch_size, seq_len)
        x = x[:, :int(lengths.max())]
        out = self.embedding(x)  # -> (batch_size, seq_len, embed_dim)
        # Bỏ qua padding để hidden state cuối ứng với từ cuối của bình luận.
        out = pack_padded_sequence(out, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(out)  # -> (batch_size, seq_len, hidden_dim)
        out, hn = self.gru(out)  # -> out: (batch_size, seq_len, hidden_dim), hn: (1, batch_size, hidden_dim)

        # Lấy hidden state cuối cùng của GRU
        last_hidden = hn[-1]  # -> (batch_size, hidden_dim)

        out = self.dropout(last_hidden)
        logits = self.fc(out)  # -> (batch_size, num_classes)
        return logits


def main():
    # Ưu tiên thư mục dữ liệu đã xử lý của dự án, sau đó xét thư mục chứa script.
    splits = ("Train", "Dev", "Test")
    script_dir = Path(__file__).resolve().parent
    candidates = (Path("data/processed"), script_dir / "data/processed", script_dir)
    data_dir = next((folder for folder in candidates
                     if all((folder / f"{split}.csv").is_file() for split in splits)), None)
    if data_dir is None:
        raise FileNotFoundError("Cần đủ Train.csv, Dev.csv và Test.csv trong data/processed hoặc cạnh script.")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_texts, y_train = read_split(data_dir / "Train.csv")
    dev_texts, y_dev = read_split(data_dir / "Dev.csv")
    test_texts, y_test = read_split(data_dir / "Test.csv")
    train_tokens = [tokenize_vi(text) for text in train_texts]

    # Chỉ học từ điển trên Train để thông tin từ Dev và Test không lọt vào mô hình.
    counts = Counter(word for words in train_tokens for word in words)
    vocab = {word: i for i, (word, _) in enumerate(counts.most_common(MAX_WORDS - 2), start=2)}
    train_loader = make_loader(train_tokens, y_train, vocab, shuffle=True)
    dev_loader = make_loader([tokenize_vi(text) for text in dev_texts], y_dev, vocab)
    test_loader = make_loader([tokenize_vi(text) for text in test_texts], y_test, vocab)
    model = HybridLSTMGRU(vocab_size=len(vocab) + 2).to(device)

    # 4. Định nghĩa hàm mất mát và bộ tối ưu hóa

    criterion = nn.BCEWithLogitsLoss()  # Tích hợp sẵn Sigmoid + Binary Cross Entropy cho ổn định số học
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 15
    PATIENCE = 3
    best_val_loss = float("inf")
    patience_counter = 0

    # 5. Huấn luyện mô hình với Early Stopping

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for inputs, lengths, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # Đánh giá trên tập Dev
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, lengths, targets in dev_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, lengths)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(dev_loader.dataset)
        print(f"Epoch {epoch + 1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early Stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_hybrid_lstm_gru.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered!")
                break

    # 6. Đánh giá mô hình trên tập Test

    model.load_state_dict(torch.load("best_hybrid_lstm_gru.pt"))
    model.eval()

    all_probs = []
    with torch.no_grad():
        for inputs, lengths, _ in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs, lengths)
            probs = torch.sigmoid(logits)  # Áp dụng Sigmoid thu được xác suất 0 -> 1
            all_probs.append(probs.cpu().numpy())

    probabilities_hybrid = np.vstack(all_probs)
    predictions_hybrid = (probabilities_hybrid >= 0.5).astype(int)

    accuracy_hybrid = accuracy_score(y_test, predictions_hybrid)
    print(f"\nSubset Accuracy (Exact Match Ratio) - Hybrid PyTorch Model: {accuracy_hybrid:.4f}")
    print(
        classification_report(
            y_test,
            predictions_hybrid,
            target_names=SENTIMENTS,
            zero_division=0
        )
    )

main()
