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

    max_length = 128
    num_labels = 3
    bert = BERT()
    model = bert.create_model(max_length, num_labels)
    model.summary()

    train_example = Processing.to_bert_input(bert.tokenizer, train_data, "Title", "Headline", max_length)
    train_y_t = Cleansing.get_y(train_data, "SentimentTitle")
    train_y_hl = Cleansing.get_y(train_data, "SentimentHeadline")

    model.fit(train_example, train_y_t, epochs=30, batch_size=32, validation_split=0.2, shuffle=True)
    model.save_weights('./model/model_title.h5')

    model.fit(train_example, train_y_hl, epochs=30, batch_size=32, validation_split=0.2, shuffle=True)
    model.save_weights('./model/model_headline.h5')


if __name__ == '__main__':
    main()