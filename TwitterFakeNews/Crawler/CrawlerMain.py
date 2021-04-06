from Crawler.TweetCrawler import load_new_tweets, load_testset_accounts, insert_testset_tweets
from Utility.Twitter import get_twitter_api

api = get_twitter_api()

#---------load-collected-accounts------------------------
# # real
# print("Load real news...")
load_new_tweets(api, fake=False)
# # fake
# print("Load fake news...")
load_new_tweets(api, fake=True)

#---------load-testset-accounts--------------------------
# load accounts from tweets of POLITIFACT category 'false' and 'pants on fire'
# load_testset_accounts(api, fake=True)

# load accounts from tweets of POLITIFACT category 'true'
# load_testset_accounts(api, fake=False)

#--------load-testset-tweets-----------------------------
# insert fake news tweets
insert_testset_tweets(api, True)
# insert real news tweets
insert_testset_tweets(api, False)
