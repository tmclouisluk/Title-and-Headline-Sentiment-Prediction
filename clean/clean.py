import numpy as np
import re
import tensorflow as tf


class Cleansing(object):
    @staticmethod
    def clean_null(df):
        empty = ((df['Title'].isnull()) \
                 | (df['Headline'].isnull()) \
                 | (df['SentimentTitle'].isnull()) \
                 | (df['SentimentHeadline'].isnull()))
        df_result = df[~empty]
        df_result = df_result[~((df_result['SentimentTitle'] == "neutral") | (df_result['SentimentHeadline'] == "neutral"))]
        return df_result

    @staticmethod
    def drop_useless_cols(df):
        df = df.reset_index()
        df = df.loc[:, ['Title', 'Headline', 'SentimentTitle', 'SentimentHeadline']]
        return df

    @staticmethod
    def set_sentiment(df, label):
        def trans_sentiment(row):
            if row == "negative":
                return 0
            # elif row == "neutral":
            #     return 0.5
            else:
                return 1

        df[label] = df[label].apply(trans_sentiment)
        return df

    @staticmethod
    def remove_special_char(df):
        def remove_string_special_characters(s):
            """
            This function removes special characters from within a string.
            parameters:
                s(str): single input string.
            return:
                stripped(str): A string with special characters removed.
            """
            # Replace special character with ' '
            stripped = re.sub('[^\w\s]', '', s)
            stripped = re.sub('_', '', stripped)
            # Change any whitespace to one space
            stripped = re.sub('\s+', ' ', stripped)
            # Remove start and end white spaces
            stripped = stripped.strip()
            return stripped

        df["Title"] = df["Title"].apply(remove_string_special_characters)
        df["Headline"] = df["Headline"].apply(remove_string_special_characters)
        return df

    @staticmethod
    def get_y(df):
        #df['Total_Sentiment'] = df.apply(lambda x: (x['SentimentTitle'] + x['SentimentHeadline'])/2, axis=1)
        df = Cleansing.set_sentiment(df, 'SentimentTitle')
        df = Cleansing.set_sentiment(df, 'SentimentHeadline')
        #df = Cleansing.set_sentiment(df, 'Total_Sentiment')
        return df[['SentimentTitle', 'SentimentHeadline']].values

    @staticmethod
    def run_clean(df):
        df_result = Cleansing.clean_null(df)
        df_result = Cleansing.drop_useless_cols(df_result)
        df_result = Cleansing.remove_special_char(df_result)
        #df_result = Cleansing.set_sentiment(df_result, 'SentimentTitle')
        #df_result = Cleansing.set_sentiment(df_result, 'SentimentHeadline')

        return df_result

