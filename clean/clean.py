class Cleansing(object):
    @staticmethod
    def clean_null(df, col):
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
    def set_sentiment(df, label):
        def trans_sentiment(row):
            if row == 0:
                return 0
            elif row > 0:
                return 1
            else:
                return -1

        df[label] = df[label].apply(trans_sentiment)
        return df
