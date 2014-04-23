import codecs
import re
import sys

import wikipedia

import gevent
import gevent.monkey
gevent.monkey.patch_all()
import gevent.pool

TITLE_SKIPS = ('Template:', 'Anexo:', 'Category:', 'Wikipedia:', 'Plantilla:',
               'Portal:', 'Module:')

HEADING_RE = re.compile(r'=+.+?=+')
SENTENCE_BOUNDARY = re.compile(r'\.(?!\d)')

CONCURRENCY = 20


# Build a corpus from the articles in the specified column of the titles
# document
def build(titles_doc, column=0):
    pool = gevent.pool.Pool(CONCURRENCY)

    with codecs.open(titles_doc, 'r', encoding='utf-8') as f:
        for line in f:
            columns = line.strip().split('\t')
            title = columns[column]

            if title.startswith(TITLE_SKIPS):
                continue

            pool.spawn(process_page, title)

    pool.join()

def process_page(title):
    retry = True
    while retry:
        try:
            page = wikipedia.page(title)
        except wikipedia.DisambiguationError:
            return
        except IOError, err:
            sys.stderr.write('Retry {} ({})\n'.format(title, err))
        else:
            retry = False

    content = page.content.replace('\n', '  ')

    # Remove headings
    content = re.sub(HEADING_RE, '', content)

    for sentence in re.split(SENTENCE_BOUNDARY, content):
        print sentence.strip().lower().encode('utf-8')
    sys.stdout.flush()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Usage: python build-corpus.py <titles_doc> <column_id> [<langcode>]'
        sys.exit(1)

    if len(sys.argv) == 4:
        wikipedia.set_lang(sys.argv[3])

    build(sys.argv[1], int(sys.argv[2]))
