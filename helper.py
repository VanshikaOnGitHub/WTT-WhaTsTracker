from urlextract import URLExtract
import pandas as pd
import plotly.graph_objects as go
from collections import Counter
import nltk

nltk.download('stopwords')
from nltk.corpus import (stopwords)
from nltk import ngrams

nltk.download('punkt')
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    # fetch no. of messages
    num_messages = df.shape[0]

    # fetch no. of words
    words = 0
    for message in df['message']:
        words += len(message.split())

    # fetch no. of media items
    num_media = df[df['message'].str.contains('omitted')].shape[0]

    # fetch no. of links
    extractor = URLExtract()
    links = 0
    for message in df['message']:
        links += len(extractor.find_urls(message))

    return num_messages, words, num_media, links


def n_messages_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df['only_date'] = df['date'].dt.date
    return df.groupby(['date', 'user']).count().reset_index()[['date', 'user', 'message']]


def most_busy_users(df):
    x = df.groupby('user').count().reset_index()[['user', 'message']].sort_values(by=['message'],
                                                                                      ascending=False).head()
    # Calculate percentages and create DataFrame
    value_counts = df['user'].value_counts(normalize=True) * 100

    # Convert to DataFrame and reset index
    new_df = round(value_counts, 2).reset_index()

    # Rename columns correctly
    new_df.columns = ['name', 'percent']

    return x, new_df


def day_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df['day_name'] = df['date'].dt.day_name()
    day_activity = df['day_name'].value_counts().reset_index()

    # Rename columns
    day_activity.columns = ['day_name', 'n_messages']  # Set the column names directly

    return day_activity


def month_activity_map(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df['month'] = df['date'].dt.month_name()

    # Count occurrences of each month and create the DataFrame
    month_activity = df['month'].value_counts().reset_index()

    # Rename columns
    month_activity.columns = ['month_name', 'n_messages']  # Set the column names directly

    return month_activity


def day_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df['day_name'] = df['date'].dt.day_name()
    df['period'] = 0
    for index, row in df.iterrows():
        if row['hour'] == 23:
            df.at[index, 'period'] = str(row['hour']) + '-00'
        elif row['hour'] == 0:
            df.at[index, 'period'] = '00-' + str(row['hour'] + 1)
        else:
            df.at[index, 'period'] = str(row['hour']) + '-' + str(row['hour'] + 1)
    return df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)


def create_wordcloud(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    message_list = ''
    for message in df['message']:
        if 'omitted' in message or message == 'This message was deleted.':
            continue
        message_list += ' ' + message
    stop_words = set(stopwords.words('english'))

    # Example text
    words = message_list.split()

    # Remove the stopwords
    words = [word.lower() for word in words if word.lower() not in stop_words and len(word) > 2]
    word_freq = Counter(words)
    df_word = pd.DataFrame.from_dict(word_freq, orient='index', columns=['frequency'])
    df_word.index.name = 'word'
    #     max_frequency = df_word['frequency'].sum()
    #     df_word['frequency'] = (df_word['frequency'] / max_frequency)*100

    return df_word


def get_phrases_frequency(selected_user, df, n=2):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    message_list = ''
    for message in df['message']:
        if 'omitted' in message or message == 'This message was deleted.':
            continue
        message_list += ' ' + message
    stop_words = set(stopwords.words('english'))

    # Example text
    words = message_list.split()

    # Remove the stopwords
    words = [word.lower() for word in words if word.lower() not in stop_words and len(word) > 2]
    message_list = ' '.join(words)
    ngram_list = list(ngrams(nltk.word_tokenize(message_list), n))
    ngram_count = Counter(ngram_list)
    x = pd.DataFrame.from_dict(ngram_count, orient='index', columns=['frequency'])
    x.index = pd.Series(x.index).apply(lambda x: ' '.join(x))
    x = x.reset_index().rename(columns={'index': 'phrase'})
    x = x.sort_values('frequency', ascending=False).reset_index(drop=True)
    return x


def keyword_timeline(selected_user, df, keyword):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df['only_date'] = df['date'].dt.date
    df['keyword_count'] = 0
    for index, row in df.iterrows():
        df['keyword_count'] = df['message'].str.lower().str.count(keyword.lower())

        # Group by date and user, summing the keyword count
    return df.groupby(['date', 'user'], as_index=False)['keyword_count'].sum()



def sentiment(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    return sid_obj.polarity_scores(sentence)


def sentiment_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df['negative'] = 0
    df['neutral'] = 0
    df['positive'] = 0
    df['sentiment'] = 0

    for index, row in df.iterrows():
        sentiment_dict = sentiment(row['message'])
        df.at[index, 'negative'] = sentiment_dict['neg'] * 100
        df.at[index, 'neutral'] = sentiment_dict['neu'] * 100
        df.at[index, 'positive'] = sentiment_dict['pos'] * 100
        if sentiment_dict['compound'] >= 0.05:
            sentiment_label = "Positive"
        elif sentiment_dict['compound'] <= - 0.05:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        df.at[index, 'sentiment'] = sentiment_label
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'frequency']  # Correctly set the column names

    return df, sentiment_counts


def emoji_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    emojis = []
    for message in df['message']:
        emojis.extend([i for i in message if emoji.is_emoji(i)])

    new_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis)))).rename(
        columns={0: 'emoji', 1: 'frequency'})

    df['n_emoji'] = 0
    for index, row in df.iterrows():
        message = row['message']
        count = emoji.emoji_count(message)
        df.at[index, 'n_emoji'] = count

    return new_df, df


