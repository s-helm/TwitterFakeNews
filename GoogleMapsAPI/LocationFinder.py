import json
from urllib import request

import editdistance as editdistance
import re

from Database.DatabaseHandler import DatabaseHandler
from NLP.TextPreprocessor import TextPreprocessor


class LocationFinder:
    key = '*******'
    pre = 'https://maps.googleapis.com/maps/api/geocode/json?address='
    post = '&key='+key

    @staticmethod
    def find_physical_location(location_name):
        location_name = location_name.lower()
        location_name = location_name.replace('land of the free','')
        location_name = location_name.replace('the','')
        location_name = location_name.replace('worldwide','')
        location_name = re.sub('n\.?y\.?c\.?', 'new york city', location_name)
        location_name = re.sub('(united states of )?america', 'united states', location_name)
        location_name = re.sub('u\.?s\.?a\.?', 'united states', location_name)
        location_name = location_name.replace('united kingdom', 'uk')
        location_name = location_name.replace('republic of texas', 'texas')
        location_name = location_name.replace('european union', '')
        location_name = TextPreprocessor.remove_urls(location_name)
        url_location = location_name.replace(' ','+')

        location = dict()
        location['address'] = None
        location['country'] = None
        if url_location.replace(' ', '') != '':
            print(url_location)
            reply = request.urlopen(LocationFinder.pre+url_location+LocationFinder.post).read().decode("utf-8")

            json_reply = json.loads(reply)
            print(json_reply)
            if json_reply['status'] == 'OK':
                best_res = None
                min_edit_distance = 1000
                for res in json_reply['results']:
                    address = res['formatted_address'].lower()
                    dist = editdistance.eval(location_name, address)
                    if dist < len(address)*(3/4) or location_name in address or address in location_name:
                        if 'locality' or 'country' in res['types']:
                            if(dist < min_edit_distance):
                                min_edit_distance = dist
                                best_res = res

                if best_res is not None:
                    country = None
                    for comp in best_res['address_components']:
                        if 'country' in comp['types']:
                            country = comp['long_name']
                    if 'locality' in best_res['types']:
                        location['address'] = best_res['formatted_address']
                    location['country'] = country
                    print("Location found for '" + location_name + "': " + str(location['address']) + ", " + str(location['country']))
                else:
                    print("No location found for '" + location_name + "'")
        return location

    @staticmethod
    def get_accounts_without_physical_location():
        # query = "SELECT u.id, u.location FROM user u, "+DatabaseHandler.ACCOUNTS_JOIN_TABLE+" uc " \
        #         "WHERE lcase(u.screen_name) = lcase(uc.name) and u.physical_location is null;"
        query = "SELECT u.id, u.location FROM user u WHERE u.physical_location is null;"
        cur = DatabaseHandler.db.cursor()
        cur.execute(query)
        DatabaseHandler.db.commit()
        res = cur.fetchall()
        cur.close()
        return res

    @staticmethod
    def insert_physical_location(user_id, loc):
        query = "UPDATE user SET physical_location=%s, country=%s WHERE id=%s;"
        cur = DatabaseHandler.db.cursor()
        cur.execute(query,(loc['address'], loc['country'], user_id))
        DatabaseHandler.db.commit()
        cur.close()

    @staticmethod
    def insert_physical_locations():
        accs = LocationFinder.get_accounts_without_physical_location()
        for acc in accs:
            if acc['location'] != '':
                location = LocationFinder.find_physical_location(acc['location'])
                LocationFinder.insert_physical_location(acc['id'], location)

if __name__ == "__main__":
    LocationFinder.insert_physical_locations()
