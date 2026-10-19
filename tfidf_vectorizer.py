import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

train = pd.read_csv("data/processed/Train.csv")
dev = pd.read_csv("data/processed/Dev.csv")
test = pd.read_csv("data/processed/Test.csv")

tfidf = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train = tfidf.fit_transform(train["comment"])
X_dev = tfidf.transform(dev["comment"])
X_test = tfidf.transform(test["comment"])

print(X_train.shape, X_dev.shape, X_test.shape)