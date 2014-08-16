#!/bin/bash

# Usage: <path_to_stopwords> <path_to_tsv>
#
# Removes seeds which are English stopwords from a TSV seed collection.

# If we are reading the stopword file, just collect in `a`
FNR==NR { a[$0]; next }

# Otherwise, only print if the first field is not a stopword
(!($1 in a))
