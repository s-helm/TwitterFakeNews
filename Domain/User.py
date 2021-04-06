class User:
    'Common base class for all users'

    def __init__(self):
        self.contributors_enabled = None
        self.created_at = None
        self.default_profile = None
        self.default_profile_image = None
        self.description = None
        self.favourites_count = None
        self.followers_count = None
        self.friends_count = None
        self.geo_enabled = None
        self.has_extended_profile = None
        self.id = None
        self.is_translator = None
        self.lang = None
        self.listed_count = None
        self.location = None
        self.name = None
        self.notifications = None
        self.profile_background_color = None
        self.profile_background_image_url = None
        self.profile_background_tile = None
        self.profile_banner_url = None
        self.profile_image_url = None
        self.profile_link_color = None
        self.profile_sidebar_border_color = None
        self.profile_sidebar_fill_color = None
        self.profile_text_color = None
        self.profile_use_background_image = None
        self.protected = None
        self.screen_name = None
        self.show_all_inline_media = None
        self.statuses_count = None
        self.time_zone = None
        self.translator_type = None
        self.url = None
        self.utc_offset = None
        self.verified = None
        self.withheld_in_countries = None
        self.withheld_scope = None
