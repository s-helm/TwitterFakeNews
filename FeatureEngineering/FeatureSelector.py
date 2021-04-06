import re


def get_label():
    return "tweet__fake"


def get_group_feature():
    return "user__id"

def split_X_y(data):
    y = data[get_label()]
    X = data.drop([get_label()], axis=1)
    return X, y

def get_feature_selection(data, all=0, original=False):
    """
    selects the features 
    :param data: 
    :param all: 0: all, 1: tweet features only, 2: user features only
    :param original: if true only original features
    :return: 
    """
    tweet_features_to_remove = [
        "tweet__additional_preprocessed_text",
        "tweet__additional_preprocessed_wo_stopwords",
        "tweet__tokenized_um_url_removed",
        "tweet__place_id",
        "tweet__source",
        "tweet__created_at",
        "tweet__withheld_scope",
        # no values
        "tweet__current_user_retweet",
        "tweet__in_reply_to_status_id",
        # all 0
        "tweet__favorited",
        "tweet__quoted_status_id",
        # no values
        "tweet__withheld_in_countries",
        "tweet__in_reply_to_screen_name",
        "tweet__in_reply_to_user_id",
        "tweet__retweeted_status_id",
        "tweet__withheld_copyright",
        "tweet__text",
        # no values
        "tweet__scopes",
        # no values. Specifies if authenticated user has retweeted
        "tweet__retweeted",
        # no values. Only for authenticated user
        # no values
        "tweet__filter_level",
        # only 'en' since filtered from database
        "tweet__lang",
        "tweet__entities_id",
        "tweet__location_id",
        "tweet__user_id",
        "tweet__fake",
        'tweet__is_quoted_status',
        'tweet__is_quote_status',
        "tweet__is_reply_to_status",
        "tweet__is_retweeted_status",
        'tweet__sent_tokenized_text',
        'tweet__pos_tags',
        'tweet__tokenized_text',
        'tweet__unicode_emojis',
        'tweet__ascii_emojis',
        'tweet__favorite_count',
        'tweet__retweet_count',
        'tweet__id',
        'tweet__key_id',
        # zero variance
        'tweet__url_only',
        'tweet__is_withheld_copyright',
        'tweet__no_text',
        'tweet__year',
        'tweet__possibly_sensitive_news',
        'tweet__nr_tokens',
        'tweet__day_of_year',
        'tweet__created_days_ago'
    ]

    user_features_to_remove = [
        "user__url_top_level_domain",
        "user__country",
        "user__physical_location",
        "user__screen_name",
        "user__profile_sidebar_border_color",
        "user__location",
        "user__profile_banner_url",
        # only 0 values
        "user__is_translator",
        "user__time_zone",
        "user__url",
        "user__profile_background_color",
        # no values
        "user__withheld_scope",
        # no values
        "user__withheld_in_countries",
        "user__name",
        "user__profile_link_color",
        "user__profile_text_color",
        "user__profile_background_image_url",
        "user__description",
        # no values
        "user__show_all_inline_media",
        # only 0 values
        "user__protected",
        # only 0 (Twitter say: rarly true)
        "user__contributors_enabled",
        "user__profile_image_url",
        "user__translator_type",
        "user__profile_sidebar_fill_color",
        # negative values
        "user__utc_offset",
        "user__created_at",
        # only 0 values
        "user__default_profile_image",
        # only 0 values
        "user__notifications",
        "user__lang",
        "user__id",
        # zero variance
        'user__has_default_profile_after_two_month',
        'user__has_url',
        'user__is_english',
        'user__more_than_50_tweets']

    data_columns = list(data.columns.values)

    select = list()
    if original:
        if all != 2:
            select.extend(get_original_tweet_features())
        if all != 1:
            select.extend(get_original_user_features())
        return select
    else:
        for col in data_columns:
            if not re.match("Unnamed.*", col):
                if all == 0:
                    if col not in tweet_features_to_remove and col not in user_features_to_remove:
                        select.append(col)
                elif all == 1:
                    if col not in tweet_features_to_remove and not re.match("user__.*", col) and col != "tweet__tf_idf_sum_grouped_by_user":
                        select.append(col)
                elif all == 2:
                    if col not in user_features_to_remove and not re.match("tweet__", col):
                        select.append(col)
        # if all == 1:
        #     select.append("user__id")
        # elif all == 2:
        #     select.append("tweet__fake")

        return select



def get_original_tweet_features():
    return [
            "tweet__retweet_count",
            # "tweet__place_id",
            # "tweet__source",
            # "tweet__created_at",
            # "tweet__withheld_scope",
            # no values
            # "tweet__current_user_retweet",
            # "tweet__in_reply_to_status_id",
            # all 0
            # "tweet__favorited",
            # "tweet__quoted_status_id",
            # no values
            # "tweet__withheld_in_countries",
            # "tweet__in_reply_to_screen_name",
            # "tweet__in_reply_to_user_id",
            "tweet__possibly_sensitive",
            # "tweet__retweeted_status_id",
            # "tweet__withheld_copyright",
            # "tweet__text",
            "tweet__favorite_count",
            # no values
            # "tweet__scopes",
            "tweet__truncated",
            # no values. Specifies if authenticated user has retweeted
            # "tweet__retweeted",
            # no values. Only for authenticated user
            "tweet__is_quote_status",
            # no values
            # "tweet__filter_level",
            # only 'en' since filtered from database
            # "tweet__lang",
            # "tweet__entities_id",
            # "tweet__location_id",
            # "tweet__user_id"
            ]

def get_original_user_features():
    return [
        "user__geo_enabled",
        # "user__screen_name",
        # "user__profile_sidebar_border_color",
        # "user__location",
        # "user__profile_banner_url",
        # only 0 values
        # "user__is_translator",
        "user__statuses_count",
        "user__profile_use_background_image",
        # "user__time_zone",
        "user__profile_background_tile",
        # "user__url",
        # "user__profile_background_color",
        "user__listed_count",
        # no values
        # "user__withheld_scope",
        "user__followers_count",
        # no values
        # "user__withheld_in_countries",
        # "user__name",
        # "user__profile_link_color",
        "user__friends_count",
        # "user__profile_text_color",
        # "user__profile_background_image_url",
        # "user__description",
        "user__has_extended_profile",
        # no values
        # "user__show_all_inline_media",
        # only 0 values
        # "user__protected",
        # only 0 (Twitter say: rarly true)
        # "user__contributors_enabled",
        # "user__profile_image_url",
        # "user__translator_type",
        # "user__profile_sidebar_fill_color",
        # negative values
        # "user__utc_offset",
        "user__verified",
        "user__default_profile",
        # "user__created_at",
        # only 0 values
        # "user__default_profile_image",
        "user__favourites_count",
        # only 0 values
        # "user__notifications",
        # "user__lang"
    ]

def get_mixed_features():
    return ['user__profile_text_color',
            'user__profile_sidebar_fill_color',
            'user__profile_link_color',
            'user__profile_sidebar_border_color',
            'user__profile_background_color']

def get_features_to_col_normalize(data):
    """
    returns all features which should be normalized based on the column
    :param data: 
    :return: 
    """
    cols_norm = list()
    cols_not_norm = list()
    for col in data.columns.values:
        if "tweet__fake" not in col and "user__id" not in col and "d2v" not in col and "pos_trigram" not in col and "topic_hdp" not in col and "tweet__topic" not in col:
            cols_norm.append(col)
        else:
            cols_not_norm.append(col)
    return cols_norm, cols_not_norm

