import tensorflow_hub as hub
import tensorflow as tf
import bert


class BERT(object):
    BERT_MODEL_HUB = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"

    def __init__(self):
        self.bert_layer = hub.KerasLayer(self.BERT_MODEL_HUB, trainable=True)
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

        x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
        x = tf.keras.layers.Dropout(0.2)(x)
        out = tf.keras.layers.Dense(num_labels, activation="sigmoid", name="dense_output")(x)

        model = tf.keras.models.Model(
            inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                      metrics=['accuracy'])

        return model
