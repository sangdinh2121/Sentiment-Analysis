import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Đọc dữ liệu đã qua tiền xử lý
train = pd.read_csv("data/processed/Train.csv")
dev = pd.read_csv("data/processed/Dev.csv")
test = pd.read_csv("data/processed/Test.csv")

# Khởi tạo TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    sublinear_tf=True,
    max_features=10000  # Giữ lại 10.000 từ vựng quan trọng nhất để chạy nhanh và nhẹ
)

# Chuyển đổi text sang ma trận đặc trưng
X_train = tfidf.fit_transform(train["comment"].astype(str))
X_dev = tfidf.transform(dev["comment"].astype(str))
X_test = tfidf.transform(test["comment"].astype(str))

# In kích thước ma trận
print("Kích thước ma trận TF-IDF:")
print(f"Train: {X_train.shape}, Dev: {X_dev.shape}, Test: {X_test.shape}")


# HUẤN LUYỆN VÀ ĐÁNH GIÁ NAIVE BAYES

# 1. Lấy cột nhãn cảm xúc y (Negative, Neutral, Positive) từ file đã tiền xử lý
y_train = train["sentiment"]
y_dev = dev["sentiment"]
y_test = test["sentiment"]

# 2. Khởi tạo mô hình Multinomial Naive Bayes và huấn luyện trên X_train, y_train
print("\n--- Đang huấn luyện Naive Bayes ---")
model = MultinomialNB(alpha=0.5)
model.fit(X_train, y_train)

# 3. Đánh giá thử trên tập Dev (Validation)
y_dev_pred = model.predict(X_dev)
print(f"Độ chính xác trên tập DEV: {accuracy_score(y_dev, y_dev_pred) * 100:.2f}%")

# 4. Dự đoán và đánh giá toàn diện trên tập TEST
y_test_pred = model.predict(X_test)

print("===== KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST =====")
print(f"Độ chính xác (Accuracy): {accuracy_score(y_test, y_test_pred) * 100:.2f}%")

print("\nBáo cáo chi tiết (Precision, Recall, F1-score theo từng nhãn):")
print(classification_report(y_test, y_test_pred, digits=4))

print("Ma trận nhầm lẫn (Confusion Matrix):")
labels = ["Negative", "Neutral", "Positive"]
cm = confusion_matrix(y_test, y_test_pred, labels=labels)
print(pd.DataFrame(cm, index=[f"Thực tế_{l}" for l in labels], columns=[f"Dự đoán_{l}" for l in labels]))


# === THỬ NGHIỆM DỰ ĐOÁN VỚI CÂU REVIEW MỚI BẤT KỲ ===
def predict_review(text):
    """Hàm nhập một câu review bất kỳ và in kết quả dự đoán."""
    text_vec = tfidf.transform([str(text)])
    pred_label = model.predict(text_vec)[0]
    probs = model.predict_proba(text_vec)[0]
    
    print(f"\n[Test review]: \"{text}\"")
    print(f"-> Dự đoán: {pred_label}")
    print(f"-> Xác suất: Negative: {probs[0]*100:.1f}% | Neutral: {probs[1]*100:.1f}% | Positive: {probs[2]*100:.1f}%")

# Test thử 3 câu mẫu
predict_review("Máy xài rất tốt, pin khỏe, màn hình đẹp, nhân viên nhiệt tình")
predict_review("Sản phẩm quá tệ, mới mua 2 ngày đã bị sập nguồn, giật lag")
predict_review("Điện thoại dùng tạm được trong tầm giá, vân tay hơi chậm")
