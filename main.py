# -*- coding: utf-8 -*-
import datetime
import pandas as pd
import numpy as np
import pylab as pl
import mpl_toolkits.basemap as bm
import twitter
import dateutil
import csv
import os
import json
from geopy.geocoders import GeoNames
from iso3166 import countries
import time
import pickle
import random
import math
import sklearn.preprocessing as sp


def log(msg):
    with open("log.txt", 'a') as lf:
        str = time.strftime("%d.%m.%Y %H:%M ") + msg
        print str
        print >>lf, str


def log_info(msg):
    msg = "INFO:   " + msg
    log(msg)


def log_warn(msg):
    msg = "WARN:   " + msg
    log(msg)


def get_train_set():
    TRAINING_SET_URL = "twitter_train.txt"
    return pd.read_csv(TRAINING_SET_URL, sep=",", header=0, names=["uid", "cat"])


def get_example_set():
    EXAMPLE_SET_URL = "twitter_example.txt"
    return pd.read_csv(EXAMPLE_SET_URL, sep=",", header=0, names=["uid", "cat"])


def get_users():
    df_users_train = get_train_set()
    df_users_example = get_example_set()
    df_users_example['cat'] = None
    res = pd.concat([df_users_train, df_users_example])
    return res

# insert keys here
def get_twitter_api():
    CONSUMER_KEY = "SiwaBFBz0zRQZd1PAed87hu7P"
    CONSUMER_SECRET = "46cfhnKYdFEWvBRPI91giPmUHLbJXt3hdJINDVbPxtTHExfMFL"
    ACCESS_TOKEN_KEY = "1106605230-GWfc9XsXM53RaCqPjGClJzZeknjbD38g8rXwjRL"
    ACCESS_TOKEN_SECRET = "9B9QVWvoUiNYRXkQZf3xXdjBvsH9AtL6sruY0XXqPN2u0"

    return twitter.Api(consumer_key=CONSUMER_KEY,
                       consumer_secret=CONSUMER_SECRET,
                       access_token_key=ACCESS_TOKEN_KEY,
                       access_token_secret=ACCESS_TOKEN_SECRET)

# # Compute the distribution of the target variable
# counts, bins = np.histogram(df_users_train["cat"], bins=[0,1,2])
# # Plot the distribution
# pl.figure(figsize=(6,6))
# pl.bar(bins[:-1], counts, width=0.5, alpha=0.4)
# pl.xticks(bins[:-1] + 0.3, ["negative", "positive"])
# pl.xlim(bins[0] - 0.5, bins[-1])
# pl.ylabel("Number of users")
# pl.title("Target variable distribution")
# pl.show()


# note that you need to install geopy and iso3166 packages
class GeoFinder(object):
    PICKLED_GEO = "geomappings.pkl"
    PICKLED_GEO_BLACKLIST = "geomappings_blacklist.pkl"
    GEO_USER_NAME = "arssher"

    def __init__(self):
        self.geolocator = GeoNames(username=self.GEO_USER_NAME)
        self.geomappings = {}
        self.geomappings_blacklist = []
        if os.path.isfile(self.PICKLED_GEO):
            with open(self.PICKLED_GEO, 'rb') as f:
                self.geomappings = pickle.load(f)
        if os.path.isfile(self.PICKLED_GEO_BLACKLIST):
            with open(self.PICKLED_GEO_BLACKLIST, 'rb') as f:
                self.geomappings_blacklist = pickle.load(f)
        log_info("known geomappings size: %s " % len(self.geomappings))
        log_info("known geo blacklist size: %s " % len(self.geomappings_blacklist))

    def __del__(self):
        with open(self.PICKLED_GEO, 'wb') as f:
            pickle.dump(self.geomappings, f, pickle.HIGHEST_PROTOCOL)
        with open(self.PICKLED_GEO_BLACKLIST, 'wb') as f:
            pickle.dump(self.geomappings_blacklist, f, pickle.HIGHEST_PROTOCOL)

    def get_location(self, location_string):
        if location_string in self.geomappings:
            return self.geomappings[location_string]
        elif location_string in self.geomappings_blacklist:
            return (0, 0, "", 0)
        else:
            location = self.geolocator.geocode(location_string, exactly_one=True, timeout=60)
            if location and u'countryCode' in location.raw:
                cc_alphabet = location.raw[u'countryCode'].encode('utf_8')
                cc_numeric = int(countries.get(cc_alphabet).numeric)
                res = (location.latitude, location.longitude, location.raw[u'countryName'].encode('utf_8'), cc_numeric)
                self.geomappings[location_string] = res
                if len(self.geomappings) % 200 == 0:
                    log_info("Geomappings size now %s" % len(self.geomappings))
                return res
            else:
                self.geomappings_blacklist.append(location_string)
                log_warn("Failed to get location for string %s" % location_string.encode('utf_8'))
                return (0, 0, "", 0)

geofinder = GeoFinder()
def get_coordinates_by_location(location_string):
    """
    This function gets geographic coordinates and city name
    form external web service GeoNames using 'location' string.

    NOTE: the returned value is FAKE. It's only used to show
    NOTE: correct output format.
    """
    return geofinder.get_location(location_string)


ts_parser = lambda date_str: dateutil.parser.parse(date_str) if pd.notnull(date_str) else None
def twitter_user_to_dataframe_record(user):
    dt = ts_parser(user.created_at)
    record = {
        "uid": user.id,
        "name": user.name,
        "screen_name": user.screen_name,
        "created_at": dt.strftime("%Y-%m") if dt else dt,
        "followers_count": user.followers_count,
        "friends_count": user.friends_count,
        "statuses_count": user.statuses_count,
        "favourites_count": user.favourites_count,
        "listed_count": user.listed_count,
        "verified": user.verified
    }

    if user.description is not None and user.description.strip() != "":
        record["description"] = user.description

    if user.location is not None and user.location.strip() != "":
        record["location"] = user.location
        # I have added country_code here
        location_tuple = get_coordinates_by_location(user.location)
        if location_tuple[2] != "":
            record["lat"], record["lon"], record["country"], record["country_code"] = get_coordinates_by_location(user.location)

    return record

f, processed_users, user_records = None, [], []  # for compatibility with notebook
def get_user_records(df, f=f, processed_users=processed_users, user_records=user_records):
    api = get_twitter_api()
    ids = df['uid'].values
    log_info("Total ids: %s" % ids.size)
    log_info("Processed ids: %s" % len(processed_users))
    unprocessed_ids = np.array(filter(lambda user_id: user_id not in processed_users, ids))
    log_info("Unprocessed ids: %s" % unprocessed_ids.size)
    i = 0
    batch = unprocessed_ids[i:i+100]
    res = []
    for user_json_dict in user_records:
        res.append(twitter.User.NewFromJsonDict(user_json_dict))
    while batch.size > 0:
        log_info(str(batch.tolist()))
        usrs_batch = []
        try:
            usrs_batch = api.UsersLookup(user_id=batch.tolist())
        except twitter.TwitterError as e:
            log_warn("Exception raised while getting info for ids %s" % str(batch))
            log_warn(str(e))
        for usr in usrs_batch:
            print >>f, usr.AsJsonString()
            res.append(usr)
        log_info("%s ids handled, %s left" % (i + batch.size, unprocessed_ids.size - i - batch.size))
        i += 100
        batch = unprocessed_ids[i:i+100]
        time.sleep(30)  # to avoid ban from twitter
    return map(lambda u: twitter_user_to_dataframe_record(u), res)


# download data and save as csv and pckl
def collect_data():
    ts_parser = lambda date_str: datetime.datetime.strptime(date_str, "%Y-%m") if pd.notnull(date_str) and date_str else None
    if os.path.isfile("hw1_out.csv"):
        return pd.read_csv("hw1_out.csv", sep="\t", encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC, converters={"created_at": ts_parser})
    if os.path.isfile("df_full.pckl"):
        return pd.read_pickle("df_full.pckl")

    user_records = []
    tmp_file_name = 'tmp_user_records.txt'
    if os.path.exists(tmp_file_name):
        with open(tmp_file_name) as f:
            for line in f:
                try:
                    user_records.append(json.loads(line))
                except:
                    continue

    processed_users = set()
    for r in user_records:
        processed_users.add(r['id'])

    with open(tmp_file_name, 'a') as f:
        user_records = get_user_records(get_users(), f, processed_users, user_records)

    print "Creating data frame from loaded data"
    df_records = pd.DataFrame(user_records, columns=["uid", "name", "screen_name", "description", "verified", "location", "lat", "lon", "country", "country_code", "created_at", "followers_count", "friends_count", "statuses_count", "favourites_count", "listed_count"])
    print "Merging data frame with the training set"
    df_full = pd.merge(get_users(), df_records, on="uid", how="left")
    print "Finished building data frame"
    OUT_FILE_PATH = "hw1_out.csv"
    print "Saving output data frame to %s" % OUT_FILE_PATH
    df_full.to_csv(OUT_FILE_PATH, sep="\t", index=False, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)
    df_full.to_pickle("df_full.pckl")
    return df_full


def count_users(grouped):
    """
    Counts number of positive and negative users
    created at each date.

    Returns:
        count_pos -- 1D numpy array with the counts of positive users created at each date
        count_neg -- 1D numpy array with the counts of negative users created at each date
        dts -- a list of date strings, e.g. ['2014-10', '2014-11', ...]
    """
    dts = []
    count_pos, count_neg = np.zeros(len(grouped)), np.zeros(len(grouped))
    # your code here
    positive_series = grouped.apply(lambda g: g[g['cat'] == 1]).groupby('created_at').size()
    negative_series = grouped.apply(lambda g: g[g['cat'] == 0]).groupby('created_at').size()
    united_index = positive_series.index.append(negative_series.index)
    positive_series = positive_series.reindex(united_index, fill_value=0)
    negative_series = negative_series.reindex(united_index, fill_value=0)

    return positive_series.values, negative_series.values, united_index.values.tolist()


def draw_by_registration_year(df_full):
    grouped = df_full.groupby(map(lambda dt: dt if pd.notnull(dt) else "NA", df_full["created_at"]))
    count_pos, count_neg, dts = count_users(grouped)

    fraction_pos = count_pos / (count_pos + count_neg + 1e-10)
    fraction_neg = 1 - fraction_pos

    sort_ind = np.argsort(dts)

    pl.figure(figsize=(20, 3))
    pl.bar(np.arange(len(dts)), fraction_pos[sort_ind], width=1.0, color='red', alpha=0.6, linewidth=0, label="Positive")
    pl.bar(np.arange(len(dts)), fraction_neg[sort_ind], bottom=fraction_pos[sort_ind], width=1.0, color='green', alpha=0.6, linewidth=0, label="Negative")
    pl.xticks(np.arange(len(dts)) + 0.4, sorted(dts), rotation=90)
    pl.title("Class distribution by account creation month")
    pl.xlim(0, len(dts))
    pl.legend()
    pl.show()


def draw_on_map(df_full):
    pl.figure(figsize=(20,12))

    m = bm.Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='c')

    m.drawcountries(linewidth=0.2)
    m.fillcontinents(color='lavender', lake_color='#000040', zorder=0)
    m.drawmapboundary(linewidth=0.2, fill_color='#000040')
    m.drawparallels(np.arange(-90,90,30),labels=[0,0,0,0], color='white', linewidth=0.5)
    m.drawmeridians(np.arange(0,360,30),labels=[0,0,0,0], color='white', linewidth=0.5)

    def plot_points_on_map(df_full, m):
        """
        Plot points on the map. Be creative.
        """
        df_positive = df_full[(pd.notnull(df_full.lat)) & (df_full.cat == 1)]
        df_positive_grouped = df_positive.groupby(['lat', 'lon']).size()
        positive_lats, positive_lons = df_positive_grouped.reset_index()['lat'].values.tolist(), df_positive_grouped.reset_index()['lon'].values.tolist()
        # skew dots a little bit to make map more readable
        positive_lats, positive_lons = map(lambda x: x + random.uniform(-0.2, 0.2), positive_lats), map(lambda x: x + random.uniform(-0.4, 0.4), positive_lons)
        positive_freq = map(lambda x: x*1, df_positive_grouped.values.tolist())

        df_negative = df_full[(pd.notnull(df_full.lon)) & (df_full.cat == 0)]
        df_negative_grouped = df_negative.groupby(['lat', 'lon']).size()
        negative_lats, negative_lons = df_negative_grouped.reset_index()['lat'].values.tolist(), df_negative_grouped.reset_index()['lon'].values.tolist()
        negative_lats, negative_lons = map(lambda x: x + random.uniform(-0.2, 0.2), negative_lats), map(lambda x: x + random.uniform(-0.4, 0.4), negative_lons)
        negative_freq = map(lambda x: x*1, df_negative_grouped.values.tolist())

        px, py = m(positive_lons, positive_lats)
        nx, ny = m(negative_lons, negative_lats)
        m.scatter(px, py, s=positive_freq, marker='o', color='r')
        m.scatter(nx, ny, s=negative_freq, marker='o', color='g')
        return

    plot_points_on_map(df_full, m)

    pl.title("Geospatial distribution of twitter users")
    pl.legend()
    pl.show()


def read_data():
    df_users = collect_data()
    # Remove rows with users not found
    df_users = df_users[pd.notnull(df_users['name'])]
    df_users["lat"].fillna(value=0, inplace=True)
    df_users["lon"].fillna(value=0, inplace=True)
    return df_users


def create_new_features(df_users, features):
    # Introduce new features
    new_features = ["name_words", "screen_name_length", "description_length", "created_year", "country_code", "verified"]

    df_users = df_users[pd.notnull(df_users['name'])]
    df_users = df_users[pd.notnull(df_users['screen_name'])]
    df_users = df_users[pd.notnull(df_users['description'])]
    df_users = df_users[pd.notnull(df_users['created_at'])]
    df_users = df_users[pd.notnull(df_users['lat'])]
    df_users = df_users[pd.notnull(df_users['lon'])]
    df_users = df_users[pd.notnull(df_users['country_code'])]
    df_users = df_users[pd.notnull(df_users['statuses_count'])]
    df_users = df_users[pd.notnull(df_users['favourites_count'])]
    df_users = df_users[pd.notnull(df_users['listed_count'])]

    df_users["name_words"] = df_users["name"].apply(lambda name: len(name.split()))
    df_users["screen_name_length"] = df_users["screen_name"].apply(lambda screen_name: len(screen_name))
    df_users["description_length"] = df_users["description"].apply(lambda x: len(x) if pd.notnull(x) else 0)
    df_users["created_year"] = df_users["created_at"].apply(lambda created_at: created_at.year)
    df_users["verified"] = df_users["verified"].apply(lambda x: int(x) if pd.notnull(x) else 0)

    # Calculate new features and place them into data frame
    # place tour code here

    features += new_features
    return df_users, features


def find_correlated_features(x, features):
    # replace this code to find really correlated features
    for i, feature_i in enumerate(features):
        for j, feature_j in enumerate(features):
            if i < j:
                coef = np.corrcoef(x[:, i], x[:, j])[0, 1]
                if abs(coef) > 0.2:
                    print "Correlated features: %s + %s -> %.2f" % (feature_i, feature_j, coef)


def plot_two_features_scatter(feature_i, x_i, feature_j, x_j, y):
    i_positive = x_i[(y != 0)]
    i_negative = x_i[(y == 0)]
    j_positive = x_j[(y != 0)]
    j_negative = x_j[(y == 0)]

    # pl.gca().axes.yaxis.set_ticklabels([])
    # pl.gca().axes.xaxis.set_ticklabels([])

    pl.scatter(j_positive, i_positive, s=0.1, color='r', alpha=0.3)
    pl.scatter(j_negative, i_negative, s=0.5, color='g', alpha=0.3)
    pl.xlabel(feature_j)
    pl.ylabel(feature_i)


def plot_feature_histogram(feature_name, x_i, y):
    positive_x = x_i[(y != 0)]
    negative_x = x_i[(y == 0)]

    # print feature_name, positive_x
    # pl.gca().axes.yaxis.set_ticklabels([])
    # pl.gca().axes.xaxis.set_ticklabels([])
    pl.hist([positive_x, negative_x], bins=20, color=['r', 'g'],  alpha=0.3, label=['positive', 'negative'])
    pl.xlabel(feature_name)


def plot_dataset(x, y, features):
    pl.subplots_adjust(hspace=0.4, wspace=0.4)
    for i, feature_i in enumerate(features):
        for j, feature_j in enumerate(features):
            pl.subplot(len(features), len(features), i * len(features) + j + 1)
            if i != j:
                plot_two_features_scatter(feature_i, x[:, i], feature_j, x[:, j], y)
            else:
                plot_feature_histogram(feature_i, x[:, i], y)

    pl.show()


def log_transform_features(data, features, transformed_features):
    def transform_row(row):
        # return map(lambda x: math.log(x[0] + 1) if (x[1] in transform_indexes) else x[0], enumerate(row))
        vfunc = np.vectorize(lambda x, i: math.log(x + 1) if (i in transform_indexes) else x)
        res = vfunc(row, range(len(row)))
        return res

    transform_indexes = [i for i, f in enumerate(features) if f in transformed_features]
    return np.apply_along_axis(transform_row, axis=1, arr=data)


def investigate_features(df_users):
    features = ["lat", "lon", "followers_count", "friends_count", "statuses_count", "favourites_count", "listed_count"]
    df_users, features = create_new_features(df_users, features)

    x = df_users[pd.notnull(df_users.cat)][features].values
    y = df_users[pd.notnull(df_users.cat)]["cat"].values

    find_correlated_features(x, features)

    geo_features_new = ["lat", "lon", "country_code"]
    geo_features = [f for f in geo_features_new if f in features]
    geo_feature_ind = [i for i, f in enumerate(features) if f in geo_features]
    plot_dataset(x[:, geo_feature_ind], y, geo_features)

    social_features_new = ["followers_count", "friends_count", "statuses_count", "favourites_count", "listed_count", "created_year", "verified"]
    social_features = [f for f in social_features_new if f in features]
    social_feature_ind = [i for i, f in enumerate(features) if f in social_features]
    print "Features: " + str(features)
    print "Social features: " + str(social_features)
    print "Social features ind: " + str(social_feature_ind)
    plot_dataset(x[:, social_feature_ind], y, social_features)

    transformed_features = ["followers_count", "friends_count", "statuses_count", "favourites_count", "listed_count"]
    x = log_transform_features(x, features, transformed_features)
    plot_dataset(x[:, social_feature_ind], y, social_features)

    selected_features = ["followers_count", "friends_count", "statuses_count", "favourites_count",
                         "listed_count", "name_words", "screen_name_length", "description_length", "created_year"]

    x_1 = df_users[selected_features].values
    y = df_users["cat"].values

    # x_1 = x[:, selected_features_ind]
    # Replace nan with 0-s
    # Is there a smarter way?
    x_1[np.isnan(x_1)] = 0  # well, isn't boolean indexing smart enough? we can also do x_1 = np.nun_to_num(x_1)

    # custom normalization
    # x_min = x_1.min(axis=0)
    # x_max = x_1.max(axis=0)
    # x_new = (x_1 - x_min) / (x_max - x_min)

    # sklearn normalization
    x_new = sp.normalize(x_1)
    df_out = pd.DataFrame(data=x_new, index=df_users["uid"], columns=[f for f in selected_features])
    df_out.to_csv("hw2_out_sknorm.csv", sep="\t")

if __name__ == "__main__":
    df_users = read_data()
    investigate_features(df_users)

