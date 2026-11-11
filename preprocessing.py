import pandas as pd
import re
import unicodedata
from pathlib import Path

# === Thư viện tách từ ghép tiếng Việt pyvi ===
try:
    from pyvi import ViTokenizer
except ImportError:
    print("Vui lòng cài đặt pyvi bằng lệnh: pip install pyvi")

input_dir = Path("data/raw")
output_dir = Path("data/processed")

output_dir.mkdir(parents=True, exist_ok=True)

files = ["Train.csv", "Dev.csv", "Test.csv"]

# === Từ điển sửa từ viết tắt / teencode thường gặp trong review điện thoại ===
TEENCODE_DICT = {
    # Phủ định
    "k": "không", "ko": "không", "kh": "không", "khong": "không", "khg": "không",
    "hok": "không", "hong": "không", "hông": "không",
    # Viết tắt
    "dc": "được", "đc": "được", "dk": "được", "đk": "được",
    "dt": "điện thoại", "dthoai": "điện thoại", "đt": "điện thoại", "sp": "sản phẩm",
    "tgdd": "thế giới di động", "nv": "nhân viên", "tks": "cảm ơn", "tk": "cảm ơn",
    "bt": "bình thường", "bth": "bình thường",
    # Khen / Chê
    "ok": "tốt", "oke": "tốt", "oki": "tốt", "okay": "tốt",
    "pin trâu": "pin khỏe", "vcl": "rất tệ", "vl": "rất tệ", "chán": "thất vọng",
    "mượt": "mượt mà", "lag": "giật", "sạc": "sạc", "sac": "sạc"
}

def clean_text(text):
    if pd.isna(text):
        return ""

    # Chuẩn hóa các loại unicode thành chung 1 kiểu
    text = unicodedata.normalize("NFC", str(text))

    # === Chuyển toàn bộ về chữ thường ===
    text = text.lower()

    # === Xóa đường link web, email, số điện thoại ===
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b\d{10,11}\b", " ", text)

    # === Rút gọn chữ kéo dài (ngonnn -> ngon, đẹpppp -> đẹp) ===
    text = re.sub(r'([a-zà-ỹ])\1{2,}', r'\1', text)

    # === Chuẩn hóa từ viết tắt / teencode ===
    words = text.split()
    words = [TEENCODE_DICT.get(w, w) for w in words]
    text = " ".join(words)

    # === Xóa ký tự đặc biệt, icon emoji (chỉ giữ lại chữ và số) ===
    text = re.sub(r"[^\w\s\d]", " ", text)

    # === Tách từ ghép tiếng Việt bằng pyvi (vd: "điện thoại" -> "điện_thoại") ===
    try:
        text = ViTokenizer.tokenize(text)
    except:
        pass

    # Gộp các khoảng trắng liên tiếp
    text = re.sub(r"\s+", " ", text)

    # Xóa khoảng trắng ở đầu và cuối
    text = text.strip()

    return text

# === Hàm gán nhãn 3 lớp (Positive, Neutral, Negative) từ số sao n_star ===
def convert_star_to_sentiment(n_star):
    try:
        star = int(float(n_star))
        if star <= 2:
            return "Negative"  # 1, 2 sao: Tiêu cực
        elif star == 3:
            return "Neutral"   # 3 sao: Trung tính
        else:
            return "Positive"  # 4, 5 sao: Tích cực
    except:
        return "Neutral"

# Vòng lặp xử lý từng file
for filename in files:
    df = pd.read_csv(input_dir / filename)

    print("------------------------------------------")
    print("File:", filename)
    print("Kích thước ban đầu:", df.shape)
    print("Giá trị thiếu:", df.isnull().sum().sum())
    print("Dòng trùng hoàn toàn:", df.duplicated().sum())

    # Xóa dòng bị thiếu comment hoặc thiếu số sao
    df = df.dropna(subset=["comment", "n_star"])

    # === Xóa các comment bị trùng lặp ===
    df = df.drop_duplicates(subset=["comment"])

    # === Tạo cột nhãn cảm xúc sentiment (Negative, Neutral, Positive) ===
    df["sentiment"] = df["n_star"].apply(convert_star_to_sentiment)

    # Chuẩn hóa cột comment
    df["comment"] = df["comment"].apply(clean_text)

    # === Xóa các dòng mà sau khi làm sạch không còn chữ nào ===
    df = df[df["comment"].str.len() > 0]

    output_path = output_dir / filename

    # Lưu ra file processed
    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("Kích thước sau xử lý:", df.shape)
    print("Phân bố nhãn cảm xúc:")
    print(df["sentiment"].value_counts())
    print("Đã lưu tại:", output_path)
