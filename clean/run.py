
MAX_LENGTH = 30


def clean_null(df):
    empty = ((df['Title'].isnull()) \
                   | (df['Headline'].isnull()) \
                   | (df['SentimentTitle'].isnull()) \
                   | (df['SentimentHeadline'].isnull()))
    df_result = df[~empty]
    return df_result


def trim_title(df):
    df = df[~(df.Title.apply(lambda x: len(x)) > MAX_LENGTH)]
    df = df[~(df.Headline.apply(lambda x: len(x)) > MAX_LENGTH)]
    return df


def drop_useless_cols(df):
    df = df.reset_index()
    df = df.loc[:, ['Title', 'Headline', 'SentimentTitle', 'SentimentHeadline']]
    return df


def run(df):
    df = clean_null(df)
    df = trim_title(df)
    return df
