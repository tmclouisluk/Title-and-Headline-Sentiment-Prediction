import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    # Load data
    train_path = "./data/train_file.csv"
    test_path = "./data/test_file.csv"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    train, val = train_test_split(train_data, test_size=0.2, random_state=420)



if __name__ == '__main__':
    main()