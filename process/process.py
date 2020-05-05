import bert


class Processing(object):
    @staticmethod
    def to_bert_example(df, col_a, col_b, col_label):
        return df.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                   text_a=x[col_a],
                                                                   text_b=x[col_b] if col_b is not None else None,
                                                                   label=x[col_label]), axis=1)

    @staticmethod
    def to_bert_feature(bert_obj, label_list, max_seq_length, tokenizer):
        return bert.run_classifier.convert_examples_to_features(bert_obj, label_list, max_seq_length, tokenizer)