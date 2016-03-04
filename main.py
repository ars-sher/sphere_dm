# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pylab as pl
import mpl_toolkits.basemap as bm
import twitter
import requests
import datetime
import dateutil
import csv
import os
import json
from geopy.geocoders import GeoNames
import time
import pickle
import io

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
    CONSUMER_KEY = ""
    CONSUMER_SECRET = ""
    ACCESS_TOKEN_KEY = ""
    ACCESS_TOKEN_SECRET = ""

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


# note that you need to install geopy package
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

    def __del__(self):
        with open(self.PICKLED_GEO, 'wb') as f:
            pickle.dump(self.geomappings, f, pickle.HIGHEST_PROTOCOL)
        with open(self.PICKLED_GEO_BLACKLIST, 'wb') as f:
            pickle.dump(self.geomappings_blacklist, f, pickle.HIGHEST_PROTOCOL)

    def get_location(self, location_string):
        if location_string in self.geomappings:
            return self.geomappings[location_string]
        elif location_string in self.geomappings_blacklist:
            return (0, 0, "")
        else:
            location = self.geolocator.geocode(location_string, exactly_one=True, timeout=60)
            if location:
                res = (location.latitude, location.longitude, location.address.partition(",")[0])
                self.geomappings[location_string] = res
                if len(self.geomappings) % 200 == 0:
                    log_info("Geomappings size now %s" % len(self.geomappings))
                return res
            else:
                self.geomappings_blacklist.append(location_string)
                log_warn("Failed to get location for string %s" % location_string.encode('utf_8'))
                return (0, 0, "")

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
        record["lat"], record["lon"], record["country"] = get_coordinates_by_location(user.location)

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
        time.sleep(30) # to avoid ban from twitter
    return map(lambda u: twitter_user_to_dataframe_record(u), res)

    # your code here
    # some_downloaded_user = get_user_from_api
    # also write user as json line in temporary file
    # return [twitter_user_to_dataframe_record(some_downloaded_user)]


def collect_data():
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
    df_records = pd.DataFrame(user_records, columns=["uid", "name", "screen_name", "description", "verified", "location", "lat", "lon", "country", "created_at", "followers_count", "friends_count", "statuses_count", "favourites_count", "listed_count"])
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


def draw_by_map(df_full):
    pl.figure(figsize=(20,12))

    m = bm.Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='c')

    m.drawcountries(linewidth=0.2)
    m.fillcontinents(color='lavender', lake_color='#000040')
    m.drawmapboundary(linewidth=0.2, fill_color='#000040')
    m.drawparallels(np.arange(-90,90,30),labels=[0,0,0,0], color='white', linewidth=0.5)
    m.drawmeridians(np.arange(0,360,30),labels=[0,0,0,0], color='white', linewidth=0.5)

    def plot_points_on_map(df_full, m):
        """
        Plot points on the map. Be creative.
        """
        df_positive = df_full[(pd.notnull(df_full.location)) & (df_full.cat == 1)]
        positive_lats, positive_lons = df_positive['lat'].values.tolist(), df_positive['lon'].values.tolist()
        df_negative = df_full[(pd.notnull(df_full.location)) & (df_full.cat == 0)]
        negative_lats, negative_lons = df_negative['lat'].values.tolist(), df_negative['lon'].values.tolist()
        px, py = m(positive_lons, positive_lats)
        nx, ny = m(negative_lons, negative_lats)
        m.scatter(px, py, marker='*', color='r')
        m.scatter(nx, ny, marker='*', color='g')
        return

    plot_points_on_map(df_full, m)

    pl.title("Geospatial distribution of twitter users")
    pl.legend()
    pl.show()

if __name__ == "__main__":
    df_full = collect_data()
    draw_by_registration_year(df_full)
    draw_by_map(df_full)

