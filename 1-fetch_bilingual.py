import json
import sys
import urllib

import gevent
import gevent.monkey
gevent.monkey.patch_all()


RANDOM_URL = 'https://www.wikidata.org/w/api.php?format=json&action=query&list=random&rnnamespace=0&rnlimit=10'
ARTICLE_URL = 'https://www.wikidata.org/wiki/Special:EntityData/{}.json'

# Drop an article if we haven't gotten a server response within a second
ARTICLE_TIMEOUT = 1


def fetch(languages):
    handle = urllib.urlopen(RANDOM_URL)
    articles = json.load(handle)['query']['random']
    gevent.joinall([gevent.spawn(try_article, article, languages)
                    for article in articles], timeout=ARTICLE_TIMEOUT)
    sys.stdout.flush()

def try_article(article, languages):
    article_id = article['title']

    article_handle = urllib.urlopen(ARTICLE_URL.format(article_id))
    article_data = json.load(article_handle)['entities'][article_id]

    # WTFAPI
    if 'labels' not in article_data:
        return

    article_languages = article_data['labels'].keys()

    if 'en' not in article_languages:
        return

    # Skip category pages
    if article_data['labels']['en']['value'].startswith('Category:'):
        return

    match = True
    for language in languages:
        if language not in article_languages:
            match = False
            break

    if match:
        result = [article_data['labels'][language]['value']
                  for language in languages]

        output_result(result)


def output_result(result):
    print '\t'.join([string.encode('utf-8') for string in result])


if __name__ == '__main__':
    for _ in range(40):
        gevent.joinall([gevent.spawn(fetch, ['en', 'es'])
                        for _ in range(150)])
