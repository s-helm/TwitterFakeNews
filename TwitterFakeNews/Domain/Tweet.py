from datetime import datetime, MINYEAR

class Tweet:
    'Common base class for all tweets'

    def __init__(self):
        self.created_at = None
        self.current_user_retweet = None
        self.entities = None
        self.fake = None
        self.favorite_count = None
        self.favorited = None
        self.filter_level = None
        self.id = None
        self.in_reply_to_screen_name = None
        self.in_reply_to_status_id = None
        self.in_reply_to_user_id = None
        self.is_quote_status = None
        self.lang = None
        self.location = None
        self.place = None
        self.possibly_sensitive = None
        self.quoted_status_id = None
        self.quoted_status = None
        self.retweet_count = None
        self.retweeted = None
        self.retweeted_status = None
        self.scopes = None
        self.source = None
        self.text = None
        self.truncated = None
        self.user = None
        self.withheld_copyright = None
        self.withheld_in_countries = None
        self.withheld_scope = None
