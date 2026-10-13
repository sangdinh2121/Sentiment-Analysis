import pandas as pd

train = pd.read_csv("data/raw/Train.csv")
dev = pd.read_csv("data/raw/Dev.csv")
test = pd.read_csv("data/raw/Test.csv")

datasets = {
    "Train": train,
    "Dev": dev,
    "Test": test
}

for name, df in datasets.items():
    print(name)

    print("Kích thước:", df.shape)
    print("\nCác cột:")
    print(df.columns.tolist())

    print(df.head())

    print("\nGiá trị thiếu:")
    print(df.isnull().sum())

    print("\nSố dòng trùng:")
    print(df.duplicated().sum())