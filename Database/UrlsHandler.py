import pymysql

db_twitter = pymysql.connect(host='localhost', user='root', password='*******', db='twitterschema', charset='utf8mb4',
                     cursorclass=pymysql.cursors.DictCursor)

db_urls = pymysql.connect(host='localhost', user='root', password='*******', db='urls', charset='utf8mb4',
                     cursorclass=pymysql.cursors.DictCursor)


def urls_to_expand_to_urls_db():
    """clears the 'urls' db. Then inserts all urls that have not been expanded yet from the 'twitterschema' db"""
    query = "SELECT * FROM url WHERE manual_expanded_url is null;"
    insert = "INSERT INTO url(url, expanded_url, manual_expanded_url) VALUES(%s,%s,%s);"
    truncate = "TRUNCATE TABLE url"

    # clear urls database
    print("Clear urls database...")
    tr_cur = db_urls.cursor()
    tr_cur.execute(truncate)
    db_urls.commit()
    tr_cur.close()

    print("Fetch urls to expand...")
    cur = db_twitter.cursor()
    cur.execute(query)
    db_twitter.commit()
    to_expand = cur.fetchall()
    cur.close()

    to_insert = list()
    for i in to_expand:
        to_insert.append(i)

    print("Urls to expand: " + str(len(to_insert)))

    urls = list()
    for item in to_insert:
        urls.append((item['url'],item['expanded_url'],item['manual_expanded_url']))

    print("Insert urls to expand...")
    url_cur = db_urls.cursor()
    url_cur.executemany(insert, urls)
    db_urls.commit()
    url_cur.close()


def expanded_urls_to_twitter_db():
    """inserts the expanded urls from the 'urls' db into the 'twitterschema' db"""
    expanded = "SELECT * FROM url WHERE manual_expanded_url is not null;"

    url_cur = db_urls.cursor()
    url_cur.execute(expanded)
    db_urls.commit()
    expanded_urls = url_cur.fetchall()
    url_cur.close()

    insert_expanded = "UPDATE url SET expanded_url = %s, manual_expanded_url = %s WHERE url = %s;"

    to_insert = list()
    for u in expanded_urls:
        to_insert.append((u['expanded_url'],u['manual_expanded_url'],u['url']))

    twitter_cur = db_twitter.cursor()
    twitter_cur.executemany(insert_expanded, to_insert)
    db_twitter.commit()
    twitter_cur.close()

if __name__ == "__main__":
    # expanded_urls_to_twitter_db()
    urls_to_expand_to_urls_db()

