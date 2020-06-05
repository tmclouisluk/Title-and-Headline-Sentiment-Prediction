import os

import pandas as pd
from sklearn.model_selection import train_test_split

from clean.clean import Cleansing
from model.model import BERT
from process.process import Processing
import tensorflow as tf


def main():
    # Load data
    train_path = "./data/train_new.csv"
    train_data = pd.read_csv(train_path)

    train_data = Cleansing.run_clean(train_data)

    train, test_data = train_test_split(train_data, test_size=0.2, random_state=420)

    max_length = 32
    num_labels = 2
    bert = BERT()
    model = bert.create_model(max_length, num_labels, learning_rate=1e-4)
    model.summary()

    train_example = Processing.to_bert_input(bert.tokenizer, train, "Title", "Headline", max_length)
    #train_y = train[["SentimentTitle", "SentimentHeadline"]].values
    train_y = Cleansing.get_y(train)

    total_epochs = 20
    checkpoint = os.path.join("./trained_model", "bert_faq.ckpt")

    model.fit(train_example, train_y, epochs=total_epochs, batch_size=32, validation_split=0.2, shuffle=True,
              callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint, save_weights_only=True, verbose=1),
                         tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
    model.save_weights('./model/model.h5')


if __name__ == '__main__':
    main()