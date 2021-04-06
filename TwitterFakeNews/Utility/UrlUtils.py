import csv
import re
from urllib.parse import urlparse
import requests


class UrlUtils:
    @staticmethod
    def expand_url(url):
        """expands an url. Returns the expanded url or the original url if expansion failed"""
        # maximum length of a shortened url
        if len(url) <= 23:
            try:

                pattern = re.compile("http://.*")
                if not pattern.match(url):
                    url = 'http://' + url
                session = requests.session()  # so connections are recycled
                resp = session.head(url, allow_redirects=True)
                session.close()
                return resp.url
                # parsed = urlparse.urlparse(url)
                # h = http.HTTPConnection(parsed.netloc)
                # h.request('HEAD', parsed.path)
                # response = h.getresponse()
                # h.close()
                # if response.status//100 == 3 and response.getheader('Location'):
                #     return response.getheader('Location')
                # else:
                #     return url
            except Exception as e:
                return None
        else:
            return url

    @staticmethod
    def extract_domain(url):
        # url = re.sub("https?://", "", url)
        # url = re.sub("www\.", "", url)
        # url = re.sub("/.*", "", url)

        import urllib.parse as urlparse

        parsed_uri = urlparse(url)
        domain = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)
        return domain

    @staticmethod
    def get_top_level_domain(url):
        pattern = re.compile('\.\w+/')
        tld_groups = pattern.search(str(url))
        tld = ""
        if tld_groups:
            tld = tld_groups.group(0)
            tld = re.sub('/','',tld)
        return tld

    @staticmethod
    def get_top_level_domain_type(tld):
        """returns top level domain type"""

        f = open('resources/tld_list.csv', 'rt')
        try:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                if row[0] == tld:
                    return row[1]
        finally:
            f.close()


