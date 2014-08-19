#!/bin/bash

# Convert GloVe VSM files to Gensim-Word2Vec files.

usage() {
    echo "Usage: glove_to_gensim.sh <glove_vectors_path> <glove_vocab_path> <out_path>" 1>&2
    exit 1
}

if ! [ $# -eq 3 ]; then
    usage
fi

python -c "
from vsm.word2vec import Word2Vec

m = Word2Vec.load_glove_format('${1}', '${2}', norm_only=True)
m.save('${3}')"
