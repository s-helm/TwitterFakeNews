import csv
import operator

from TrendingTopics.DBLocation import DBLocation


def get_prepared_woeid_map(file):
    """creates a list with {utc_offet: set(woeids)}"""
    locations = get_woeid_map(file)

    prep = dict()

    for l in locations:
        woeid = l.woeid
        tmp = {}
        for l2 in locations:
            if woeid == l2.woeid:
                utc = l2.utc_offset
                if utc not in tmp:
                    tmp[utc] = 1
                else:
                    tmp[utc] += 1
        utc_offset = max(tmp.items(), key=operator.itemgetter(1))[0]
        if utc_offset not in prep:
            prep[utc_offset] = set()
            if woeid is not None:
                prep[utc_offset].add(woeid)
        else:
            if woeid is not None:
                prep[utc_offset].add(woeid)
    return prep


def get_woeid_map(file):
    woeids = []
    f = open(file, 'rt', encoding="ISO-8859-1")
    reader = csv.reader(f, delimiter=';')
    for row in reader:
        loc = DBLocation()
        loc.name = row[0]
        loc.utc_offset = int(row[1])
        if row[2] != 'null':
            loc.woeid = int(row[2])

        woeids.append(loc)

    f.close()
    return woeids
