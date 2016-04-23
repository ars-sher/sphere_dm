# -*- coding: UTF-8 -*-

import pandas as pd
from collections import defaultdict
import json
import pickle
import numpy as np
import nltk
import os
from sklearn.feature_extraction import DictVectorizer
import time
import pprint
import twitter
import cPickle

import main
from main import log_info, log_warn


class TweetsGetter(object):
    PICKLED_TWEETS = "tweets.pkl"
    BLACK_LISTED_IDS = "black_listed_ids.pkl"
    NUMBER_OF_TWEETS_TO_GET = 200  # for each user, hardcoded. Max value for GetUserTimeline method

    def __init__(self):
        self.api = main.get_twitter_api()
        self.tweets = {}
        self.black_list_ids = []
        if os.path.isfile(self.PICKLED_TWEETS):
            with open(self.PICKLED_TWEETS, 'rb') as f:
                self.tweets = cPickle.load(f)
        if os.path.isfile(self.BLACK_LISTED_IDS):
            with open(self.BLACK_LISTED_IDS, 'rb') as f:
                self.black_list_ids = cPickle.load(f)
        self.tweets_cached = len(self.tweets)
        log_info("number of users with tweets cached: %s " % self.tweets_cached)
        log_info("number of black listed users cached: %s " % len(self.black_list_ids))

    def __del__(self):
        if len(self.tweets) > self.tweets_cached:
            log_info("Saving downloaded tweets and blacklist ids")
            with open(self.PICKLED_TWEETS, 'wb') as f:
                cPickle.dump(self.tweets, f, pickle.HIGHEST_PROTOCOL)
            with open(self.BLACK_LISTED_IDS, 'wb') as f:
                cPickle.dump(self.black_list_ids, f, pickle.HIGHEST_PROTOCOL)

    # returns None in case of problems
    def get_tweets(self, user_id):
        if user_id in self.tweets:
            return self.tweets[user_id]
        elif user_id in self.black_list_ids:
            return None
        else:
            log_info("Retrieving tweets of user with id %s..." % user_id)
            tweets = None
            while True:
                try:
                    tweets = self.api.GetUserTimeline(user_id=user_id, count=self.NUMBER_OF_TWEETS_TO_GET,
                                                      include_rts=False, exclude_replies=True)
                except twitter.TwitterError, e:
                    # skip private and deleted accounts
                    if e.message == "Not authorized." or e.message == [{u'message': u'Sorry, that page does not exist.', u'code': 34}]:
                        self.black_list_ids.append(user_id)
                        log_info("User %s added to black list" % user_id)
                        return None
                    elif e.message == [{u'message': u'Rate limit exceeded', u'code': 88}]:
                        log_warn("Rate limit exceeded exception was thrown! Trying to retrieve tweets again")
                        continue
                    else:
                        self.__del__()
                        raise e
                except IOError:
                    log_warn("IOError exception was thrown! No idea that it means, but trying to retreive tweets again")
                    continue
                finally:
                    users_loaded = len(self.tweets)
                    if users_loaded % 50 == 0:
                        log_info("%s users loaded, spilling them to disk" % users_loaded)
                        self.__del__()
                break

            self.tweets[user_id] = map(lambda tweetStatus: tweetStatus.AsDict(), tweets)
            return self.tweets[user_id]


# returns None in case of problems, or list of tweets, each as a dict, otherwise
# get_all_users_tweets is used instead
def get_user_tweets(user_id):
    pass


def get_all_users_tweets(uids):
    tweets_getter = TweetsGetter()
    uids_with_tweets = map(lambda uid: (uid, tweets_getter.get_tweets(uid)), uids)
    log_info("All tweets are loaded")
    return uids_with_tweets

# parse string to list of words, lowercasing, removing refs to other users (@usrname), reducing number of vowels,
# , filtering out http(s) links
class Wordenizer(object):
    tokenizer = nltk.tokenize.TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

    @staticmethod
    def wordenize(text):
        tokenized = Wordenizer.tokenizer.tokenize(text)
        http_filtered = filter(lambda word: not word.startswith("http"), tokenized)
        return http_filtered


def get_words(text):
    words = Wordenizer.wordenize(text)
    words = filter(lambda word: len(word) > 2, words) # filter out short strings, including punctuation
    return words


class Normalizer(object):
    lemmatizer = nltk.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords += [u'every', u'wa', u'going', u"don't", u"i'm", u"via", u"really", u"time", u'would']

    @staticmethod
    def tokenize(word):
        lemmatized = Normalizer.lemmatizer.lemmatize(word)
        return lemmatized


def get_tokens(words):
    # normalize
    words_normalized = map(lambda w: Normalizer.tokenize(w), words)
    stop_words_filtered = filter(lambda word: word not in Normalizer.stopwords, words_normalized)
    return stop_words_filtered


# accepts tweet as Status.AsDict() result dict
def get_tweet_tokens(tweet):
    assert(tweet is not None)
    words = get_words(tweet["text"])
    return get_tokens(words)


# accepts list of tweets, each as a dict, turns them into tokens and calculates frequency of each
def tweets_to_grouped_tokens(tweets):
    if tweets is None:
        return {}
    tokens = map(lambda tweet: get_tweet_tokens(tweet), tweets)
    tokens_flattened = [token for tokenlist in tokens for token in tokenlist]
    tokens_freq_dict = {}
    for token in tokens_flattened:
        if token in tokens_freq_dict:
            tokens_freq_dict[token] += 1
        else:
            tokens_freq_dict[token] = 1
    # tokens_freq_dict = {key: len(list(group)) for key, group in groupby(tokens_flattened)} # why it doesn't work?
    return tokens_freq_dict


def collect_users_tokens(df_users):
    uids = df_users["uid"].values
    uids_with_tweets = get_all_users_tweets(uids)
    uids_with_token_dict_freqs = map(lambda uid_with_tweets: (uid_with_tweets[0],
                                                              tweets_to_grouped_tokens(uid_with_tweets[1])),
                                     uids_with_tweets)
    uids = map(lambda uids_with_token_dict_freqs: uids_with_token_dict_freqs[0], uids_with_token_dict_freqs)
    dict_freqs = map(lambda uids_with_token_dict_freqs: uids_with_token_dict_freqs[1], uids_with_token_dict_freqs)
    return (uids, dict_freqs)


def get_full_token_set(users_tokens):
    tokens = set()
    for d in users_tokens:
        dtokens = d.keys()
        tokens |= set(dtokens)
    return tokens


def get_full_frequencies(users_tokens):
    freqs = {}
    for d in users_tokens:
        for token, freq in d.items():
            if token in freqs:
                freqs[token] = freqs[token] + freq
            else:
                freqs[token] = freq
    return freqs


def draw_tag_cloud(users_tokens):
    from PIL import Image
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud, ImageColorGenerator

    trump_coloring = np.array(Image.open("pics/trump.png"))

    freqs = get_full_frequencies(users_tokens)
    freq_pairs = freqs.items()
    wc = WordCloud(max_words=2000, mask=trump_coloring,
                   max_font_size=40, random_state=42)
    wc.generate_from_frequencies(freq_pairs)

    image_colors = ImageColorGenerator(trump_coloring)

    # plt.imshow(wc)
    # plt.axis("off")
    #
    # plt.figure()
    plt.imshow(wc.recolor(color_func=image_colors))
    # recolor wordcloud and show
    # we could also give color_func=image_colors directly in the constructor
    # plt.imshow(trump_coloring, cmap=plt.cm.gray)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    df_users = main.get_users()
    # df_users = df_users[:1000]

    users, users_tokens = collect_users_tokens(df_users)
    token_set = get_full_token_set(users_tokens)
    v = DictVectorizer()
    vs = v.fit_transform(users_tokens)
    np.savez("files/out_4.dat", data=vs, users=users, users_tokens=list(token_set))

    draw_tag_cloud(users_tokens)
