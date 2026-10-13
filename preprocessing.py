import pandas as pd
import re
import unicodedata
from pathlib import Path

input_dir = Path("data/raw")
output_dir = Path("data/processed")

output_dir.mkdir(parents=True, exist_ok=True)

files = ["Train.csv", "Dev.csv", "Test.csv"]

def clean_text(text):
    if pd.isna(text):
        return ""

    # Chuẩn hóa các loại unicode thành chung 1 kiểu
    text = unicodedata.normalize("NFC", text)

    # Gộp các khoảng trắng liên tiếp
    text = re.sub(r"\s+", " ", text)

    # Xóa khoảng trắng ở đầu và cuối
    text = text.strip()

    return text

for filename in files:
    df = pd.read_csv(input_dir / filename)

    print("Kích thước ban đầu:", df.shape)
    print("Giá trị thiếu:", df.isnull().sum().sum())
    print("Dòng trùng hoàn toàn:", df.duplicated().sum())

    # Chuẩn hóa cột comment
    df["comment"] = df["comment"].apply(clean_text)

    output_path = output_dir / filename

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("Kích thước sau xử lý:", df.shape)
    print("Đã lưu tại:", output_path)

