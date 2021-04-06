from datetime import timedelta, datetime
from functools import lru_cache

import pandas as pd


class TimeUtils:

    @staticmethod
    def get_time():
        return datetime.now()

    @staticmethod
    def utc_to_timezone(utc_time, tz):
        if pd.isnull(tz):
            tz=-14400
        localtime = utc_time + timedelta(hours=int(tz)/60/60)
        return localtime

    @staticmethod
    def get_utc_time():
        return datetime.utcnow()

    @staticmethod
    def time_diff_in_min(t1, t2):
        return (t1-t2).seconds/60

    @staticmethod
    @lru_cache(maxsize=None)
    def days_ago(created_at, relative_to=None):
        """returns the difference between the creation data and today in days"""
        if relative_to is None:
            now = TimeUtils.get_utc_time()
        else:
            now = datetime.strptime(relative_to, "%Y-%m-%d %H:%M")
        return (now - created_at).days

    @staticmethod
    @lru_cache(maxsize=None)
    def month_ago(created_at):
        """returns the difference between the creation data and today in days"""
        now = TimeUtils.get_utc_time()
        return (now.year - created_at.year)*12 + now.month - created_at.month

    @staticmethod
    def mysql_to_python_datetime(mysql_time):
        """converts a mysql datetime to a python datetime"""
        return datetime.strptime(mysql_time, '%Y-%m-%d %H:%M:%S')

    @staticmethod
    def is_pm(created_at, tz):
        """checks if the 'created_at' is am or pm, given the timezone (default: -14400 because most accounts are from this timezone)"""
        if pd.isnull(tz):
            tz=-14400
        local_time = TimeUtils.utc_to_timezone(created_at, tz)
        date_format = '%p'
        am_pm = datetime.strftime(local_time, date_format)
        if am_pm == 'PM':
            return True
        else:
            return False

    @staticmethod
    def hour_of_day(created_at, tz):
        """returns the hour of the datetime"""
        if tz is None:
            tz = -14400
        local_time = TimeUtils.utc_to_timezone(created_at, tz)
        return local_time.hour
