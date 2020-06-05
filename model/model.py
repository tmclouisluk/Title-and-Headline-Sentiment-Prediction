import math

import tensorflow_hub as hub
import tensorflow as tf
import bert


class BERT(object):
    BERT_MODEL_HUB = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"

    def __init__(self):
        self.bert_layer = hub.KerasLayer(self.BERT_MODEL_HUB, trainable=False)
        vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()

        self.tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)

    def create_model(self, max_seq_length, num_labels=2, learning_rate=2e-5):

        input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                               name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                            name="segment_ids")

        pooled_output, sequence_output = self.bert_layer([input_word_ids, input_mask, segment_ids])

        x = tf.keras.layers.GlobalMaxPool1D()(sequence_output)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        out = tf.keras.layers.Dense(num_labels, activation="sigmoid", name="dense_output")(x)

        model = tf.keras.models.Model(
            inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                      metrics=['accuracy', tf.keras.metrics.AUC()])

        return model

    @staticmethod
    def create_learning_rate_scheduler(max_learn_rate=5e-5, end_learn_rate=1e-7, warmup_epoch_count=10, total_epoch_count=90):
        def lr_scheduler(epoch):
            if epoch < warmup_epoch_count:
                res = (max_learn_rate / warmup_epoch_count) * (epoch + 1)
            else:
                res = max_learn_rate * math.exp(
                    math.log(end_learn_rate / max_learn_rate) * (epoch - warmup_epoch_count + 1) / (
                                total_epoch_count - warmup_epoch_count + 1))
            return float(res)

        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

        return learning_rate_scheduler
