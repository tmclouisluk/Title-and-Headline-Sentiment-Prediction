import pandas as pd
from sklearn.model_selection import train_test_split

from clean.clean import Cleansing
from model.model import BERT
from process.process import Processing


def main():
    # Load data
    train_path = "./data/train_file.csv"
    train_data = pd.read_csv(train_path)

    train_data = Cleansing.run_clean(train_data)[:100]

    max_length = 128
    num_labels = 3
    bert = BERT()
    model = bert.create_model(max_length, num_labels)
    model.summary()

    train_example = Processing.to_bert_input(bert.tokenizer, train_data, "Title", "Headline", max_length)
    train_y = Cleansing.get_y(train_data, "SentimentTitle")

    model.fit(train_example, train_y, epochs=30, batch_size=32, validation_split=0.2, shuffle=True)


if __name__ == '__main__':
    main()