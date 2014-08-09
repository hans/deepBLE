#!/bin/sh

SCRIPT=$0

usage() {
    echo "Usage: $SCRIPT <word2vec_vocab_path>

	Outputs filtered vocabulary to standard output." 1>&2
    exit 1
}

if [ -z "${1}" ]; then
    usage
fi

# 1. Remove uppercase words (NB: grep [A-Z] class includes things like
#    Á, etc.) and words with especially pernicious punctuation
grep -v '[A-Z./()@#\$%^&*_=+{}|\\’“”—]' "${1}"
