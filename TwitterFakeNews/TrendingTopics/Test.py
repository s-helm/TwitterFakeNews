import datetime

# fmt = '%Y-%m-%d %H:%M:%S %Z%z'
# key = 0
# text = "Trending topics saved at: " + datetime.datetime.now().strftime(fmt) + " UTC " + str(key / 60 / 60)
# print(text)
import pymysql

from TrendingTopics.CSVReader import get_woeid_map

db = pymysql.connect(host='localhost', user='root', password='*******', db='twitterschema', charset='utf8mb4',
                     cursorclass=pymysql.cursors.DictCursor)


def update_user_offset(list):
    for i in list:
        query = "UPDATE user " \
                "SET utc_offset = %s " \
                "WHERE utc_offset is null " \
                "AND location = %s;"

        cur = db.cursor()
        cur.execute(query, (i.utc_offset, i.name))
        db.commit()
        cur.close()

map = get_woeid_map("woeid_map.csv")

update_user_offset(map)