import numpy as np


class Processing(object):
    @staticmethod
    def create_single_input(tokenizer, sentence1, sentence2, max_seq_length):
        def get_masks(tokens, max_seq_length):
            return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))

        def get_segments(tokens, max_seq_length):
            """Segments: 0 for the first sequence, 1 for the second"""
            segments = []
            current_segment_id = 0
            for token in tokens:
                segments.append(current_segment_id)
                if token == "[SEP]":
                    current_segment_id = 1
            return segments + [0] * (max_seq_length - len(tokens))

        def get_ids(tokens, tokenizer, max_seq_length):
            """Token ids from Tokenizer vocab"""
            token_ids = tokenizer.convert_tokens_to_ids(tokens, )
            input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
            return input_ids

        stokens = tokenizer.tokenize(sentence1)
        stokens = stokens[:(int(max_seq_length*9/(9+26)) - 2)]

        stokens2 = tokenizer.tokenize(sentence2)
        stokens2 = stokens2[:(int(max_seq_length*26/(9+26)) - 1)]
        stokens = ["[CLS]"] + stokens + ["[SEP]"] + stokens2 + ["[SEP]"]

        ids = get_ids(stokens, tokenizer, max_seq_length)
        masks = get_masks(stokens, max_seq_length)
        segments = get_segments(stokens, max_seq_length)

        return ids, masks, segments

    @staticmethod
    def to_bert_input(tokenizer, data, label1, label2, max_seq_length):

        input_ids, input_masks, input_segments = [], [], []

        for row in data.itertuples():
            ids, masks, segments = Processing.create_single_input(tokenizer, getattr(row, label1), getattr(row, label2), max_seq_length)

            input_ids.append(ids)
            input_masks.append(masks)
            input_segments.append(segments)

        return [np.asarray(input_ids, dtype=np.int32),
                np.asarray(input_masks, dtype=np.int32),
                np.asarray(input_segments, dtype=np.int32)]
