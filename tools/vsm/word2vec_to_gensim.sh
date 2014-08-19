#!/bin/bash

# Convert Word2Vec binary VSM files to Gensim-Word2Vec files.

usage() {
    echo "Usage: word2vec_to_gensim.sh <word2vec_vectors_path> <word2vec_vocab_path> <out_path>" 1>&2
    exit 1
}

if ! [ $# -eq 3 ]; then
    usage
fi

python -c "
from vsm.word2vec import Word2Vec

m = Word2Vec.load_word2vec_format('${1}', fvocab='${2}', binary=True, norm_only=True)
m.save('${3}')"
