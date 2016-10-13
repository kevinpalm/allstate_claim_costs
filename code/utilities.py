import pandas as pd

def read_data():
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    return train, test

def main():
    train, test = read_data()
    print(train.head())
    print(train.columns.values)

if __name__ == '__main__':
    main()