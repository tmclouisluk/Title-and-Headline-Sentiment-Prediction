import pandas as pd
from sklearn.model_selection import train_test_split

from clean.clean import Cleansing
from model.model import BERT
from process.process import Processing
import tensorflow as tf


def main():
    # Load data
    train_path = "./data/train_file.csv"
    train_data = pd.read_csv(train_path)

    train_data = Cleansing.run_clean(train_data)

    train, test_data = train_test_split(train_data, test_size=0.2, random_state=420)

    max_length = 24
    num_labels = 3
    bert = BERT()
    model = bert.create_model(max_length, num_labels, learning_rate=3e-4)
    model.summary()

    train_example = Processing.to_bert_input(bert.tokenizer, train, "Title", "Headline", max_length)
    #train_y = train[["SentimentTitle", "SentimentHeadline"]].values
    train_y = Cleansing.get_y(train)

    model.fit(train_example, train_y, epochs=6, batch_size=8, validation_split=0.2, shuffle=True,
              callbacks=[BERT.create_learning_rate_scheduler(max_learn_rate=1e-5, end_learn_rate=1e-7,
                                                        warmup_epoch_count=20, total_epoch_count=50),
                         tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
    model.save_weights('./model/model.h5')


if __name__ == '__main__':
    main()