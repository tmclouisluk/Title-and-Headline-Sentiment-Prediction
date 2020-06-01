import numpy as np


class Cleansing(object):
    @staticmethod
    def clean_null(df):
        empty = ((df['Title'].isnull()) \
                 | (df['Headline'].isnull()) \
                 | (df['SentimentTitle'].isnull()) \
                 | (df['SentimentHeadline'].isnull()))
        df_result = df[~empty]
        return df_result

    @staticmethod
    def drop_useless_cols(df):
        df = df.reset_index()
        df = df.loc[:, ['Title', 'Headline', 'SentimentTitle', 'SentimentHeadline']]
        return df

    @staticmethod
    def run_clean(df):
        df_result = Cleansing.clean_null(df)
        df_result = Cleansing.drop_useless_cols(df_result)

        return df_result

    @staticmethod
    def get_y(df, label):
        df['Negative'] = df[label].apply(lambda x: 1 if x < -0.05 else 0)
        df['Neutral'] = df[label].apply(lambda x: 1 if -0.05 <= x <= 0.05 else 0)
        df['Positive'] = df[label].apply(lambda x: 1 if x > 0.05 else 0)

        return df[['Negative', 'Neutral', 'Positive']].values

