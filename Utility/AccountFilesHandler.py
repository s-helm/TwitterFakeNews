import csv
from collections import Counter
from tweepy.streaming import json
from Utility.JsonUtils import read_accounts


def get_all_accounts_tuples():
    fake_list = get_accounts(True)
    fake = [(a.lower(), True) for a in fake_list]
    real_list = get_accounts(False)
    real = [(a.lower(), False) for a in real_list]

    accs = list()
    accs.extend(fake)
    accs.extend(real)
    return accs

def get_user_crawled_original_collection():
    insert_collection = list()

    accs = read_accounts('fake/fake_news_accounts_opensource.json')
    for acc in accs:
        acc = acc.lower()
        insert_collection.append((acc, True, "fake_opensource"))
    accs = read_accounts('fake/satire_news_accounts_opensource.json')
    for acc in accs:
        acc = acc.lower()
        insert_collection.append((acc, True, "satire_opensource"))
    accs = read_accounts('fake/fake_news_accounts.json')
    for acc in accs:
        acc = acc.lower()
        insert_collection.append((acc, True, "fake_research"))
    accs = read_accounts('fake/parody_news_accounts.json')
    for acc in accs:
        insert_collection.append((acc, True, "parody_research"))

    all_accs = [i.lower() for i in get_accounts(False)]

    accs = read_accounts('real/opensource_real_news_accounts.json')
    for acc in accs:
        acc = acc.lower()
        insert_collection.append((acc, False, "reliable_opensource"))
    accs = read_accounts('real/dmoz_breaking_news_accounts.json')
    for acc in accs:
        acc = acc.lower()
        insert_collection.append((acc, False, "reliable_dmoz"))
    accs = read_accounts('real/dmoz_us_states_news_accounts_random_selection.json')
    for acc in accs:
        acc = acc.lower()
        insert_collection.append((acc, False, "reliable_dmoz_local"))
    accs = read_accounts('real/reliable_news_accounts_from_study.json')
    for acc in accs:
        acc = acc.lower()
        insert_collection.append((acc, False, "reliable_study"))

    return insert_collection

def get_accounts(fake):
    """returns all accounts that spread fake/real news"""
    if fake:
        fake_accounts = list()
        fake_accounts.extend(read_accounts('fake/fake_news_accounts_opensource.json'))
        fake_accounts.extend(read_accounts('fake/fake_news_accounts.json'))
        return remove_dublicates(fake_accounts)
    else:
        real_accounts = list()
        real_accounts.extend(read_accounts('real/opensource_real_news_accounts.json'))
        real_accounts.extend(read_accounts('real/dmoz_us_states_news_accounts_random_selection.json'))
        real_accounts.extend(read_accounts('real/reliable_news_accounts_from_study.json'))
        return remove_dublicates(real_accounts)


def remove_dublicates(accounts):
    accounts = set(accounts)
    res = []
    for acc in accounts:
        in_list = False
        for r in res:
            if acc.lower() == r.lower():
                in_list = True
        if not in_list:
            res.append(acc)
    return res

def inspect_bs_json():
    """inspects the the file with the sources from bs detector"""
    with open('../accounts/bs_detector_sources.json') as data_file:
        data = json.load(data_file)

    list = ['bias', 'fake', 'unreliable', 'conspiracy', 'rumor', 'clickbait', 'hate', 'junksci', 'satire', 'unknown',
            'political']

    type_map = {}

    with open('../accounts/bs_websites.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)

        tmp_list = []
        of_interest = []
        for i in data:
            type = data[i]['type']
            tmp_list.append(type)
            if (type == 'fake' or type == 'satire') and data[i]['language'] == 'en':
                of_interest.append(i)
        writer.writerow(of_interest)
        print(str(len(of_interest)) + ' pages of interest')

    print(Counter(tmp_list))


def find_and_remove_duplicates(in_csv, out_json):
    """creates a file that contains only sources from the bs detector that are not listed in any other list"""

    accounts = []

    with open(in_csv) as csvfile:
        data = csv.reader(csvfile, delimiter=";")
        col = []
        data = list(data)[1:]
        for row in data:
            col.append(list(row)[0])

    for i in col:
        i = i.lower()
        accounts.append(i)

    with open(out_json, 'w') as outfile:
        json.dump({"accounts": accounts}, outfile)
