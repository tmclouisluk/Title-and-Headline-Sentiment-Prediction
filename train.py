import pandas as pd
from sklearn.model_selection import train_test_split

from clean.clean import Cleansing
from model.model import BERT
from process.process import Processing


def main():
    # Load data
    train_path = "./data/train_file.csv"
    train_data = pd.read_csv(train_path)

    train_data = Cleansing.run_clean(train_data)

    train, test_data = train_test_split(train_data, test_size=0.2, random_state=420)

    max_length = 30
    num_labels = 2
    bert = BERT()
    model = bert.create_model(max_length, num_labels)
    model.summary()

    train_example = Processing.to_bert_input(bert.tokenizer, train, "Title", "Headline", max_length)
    train_y = train[["SentimentTitle", "SentimentHeadline"]].values

    model.fit(train_example, train_y, epochs=6, batch_size=6, validation_split=0.2, shuffle=True)
    model.save_weights('./model/model.h5')


if __name__ == '__main__':
    main()