import csv as csv
import pandas as pd



def write_data_to_CSV(result, filename):
    """writes the data to a csv file"""
    with open(filename, "w", encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=';', quoting=csv.QUOTE_NONNUMERIC, lineterminator='\n')
        # keys = [key.rstrip() for key in result[0]]

        writer.writerow(result[0])  # write headers

        for i in result:
            values = [' '.join(value.splitlines()) if isinstance(value, str) else value for key, value in i.items()]
            # values = [value for key, value in i.items()]
            writer.writerow(values)


def read_data_from_CSV(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
    return reader


def save_df_as_csv(df, file_name):
    print("Store data {}".format(df.shape))
    df.to_csv(file_name, sep=';', encoding='utf-8')


def counter_to_csv(counter, filename):
    """writes a counter so a csv file"""
    with open(filename, "w", encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        for key, count in counter:
            writer.writerow([key, count])


def load_data_from_CSV(filename):
    # f = codecs.open(filename, encoding='utf8')
    print("Load data from "+filename+"...")
    # f = open(filename)
    # f.readline()  # skip the header
    # return np.genfromtxt(filename, delimiter=';', names=True, dtype=None, comments='XXXX')
    return pd.read_csv(filename, dtype=getDTypes(), quotechar='"', delimiter=';', skipinitialspace=True, na_filter=True, na_values=['','none','null'], index_col=0)

def read_header_from_CSV(filename):
    print("Load data from "+filename+"...")

    return pd.read_csv(filename, dtype=getDTypes(), quotechar='"', delimiter=';', skipinitialspace=True, na_filter=True,
                       na_values=['', 'none', 'null'], index_col=0, nrows=1)

def getDTypes():
    dtype = {"user__geo_enabled": bool,
             "user__screen_name": str,
             "user__profile_sidebar_border_color": str,
             "tweet__retweet_count": int,
             "user__location": str,
             "tweet__place_id": int,
             "tweet__source": str,
             "user__profile_banner_url": str,
             "user__is_translator": bool,
             "user__statuses_count": int,
             "tweet__user_id": int,
             "tweet__created_at": str,
             "user__profile_use_background_image": bool,
             "user__time_zone": str,
             "tweet__withheld_scope": str,
             "user__description": str,
             "tweet__current_user_retweet": int,
             "user__profile_background_tile": bool,
             "user__url": str,
             "user__profile_background_color": str,
             "user__listed_count": int,
             "user__withheld_scope": str,
             "user__followers_count": int,
             "user__withheld_in_countries": str,
             "tweet__in_reply_to_status_id": int,
             "tweet__favorited": bool,
             "user__name": str,
             "user__profile_link_color": str,
             "user__friends_count":int,
             "tweet__quoted_status_id":int,
             "user__profile_text_color":str,
             "user__profile_background_image_url":str,
             "tweet__id":int,
             "tweet__withheld_in_countries":str,
             "tweet__in_reply_to_screen_name":str,
             "tweet__in_reply_to_user_id":int,
             "tweet__location_id":int,
             "user__has_extended_profile":bool,
             "user__show_all_inline_media":bool,
             "user__protected":bool,
             "user__contributors_enabled":bool,
             "user__profile_image_url":str,
             "user__translator_type":str,
             "tweet__entities_id":int,
             "tweet__possibly_sensitive":bool,
             "user__profile_sidebar_fill_color":str,
             "user__utc_offset":int,
             "tweet__retweeted_status_id":int,
             "user__verified":bool,
             "user__default_profile":bool,
             "tweet__withheld_copyright":str,
             "user__created_at":str,
             "user__default_profile_image":bool,
             "tweet__text":str,
             "tweet__favorite_count":int,
             "tweet__scopes":str,
             "tweet__truncated":bool,
             "user__favourites_count":int,
             "tweet__retweeted":int,
             "user__notifications":bool,
             "tweet__is_quote_status":bool,
             "tweet__filter_level":str,
             "tweet__key_id":int,
             "tweet__lang":str,
             "user__id":int,
             "tweet__fake":bool,
             "user__lang": str
}

def read_csv(filename):
    f = open(filename, 'rt')
    res = list()
    try:
        reader = csv.reader(f)
        res = [row for row in reader]

    finally:
        f.close()

    return  res

def write_to_csv(filename, data):
    f = open(filename, 'wt')
    try:
        writer = csv.writer(f, lineterminator='\n')
        for row in data:
            writer.writerow(row)
    finally:
        f.close()