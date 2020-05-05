import tensorflow_hub as hub
import tensorflow as tf
import bert


class BERT(object):
    BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

    def __init__(self):
        with tf.Graph().as_default():
            self.bert_module = hub.Module(self.BERT_MODEL_HUB)
            tokenization_info = self.bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

        self.tokenizer = bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)