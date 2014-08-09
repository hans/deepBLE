#!/bin/zsh

# Build word2vec VSMs for a whole slew of common configurations.

CORPUS="${1}"
shift

if [ -z $CORPUS ]; then
    echo 'Missing corpus argument' 1>&2
    exit 1
fi

# Extra arguments to override / extend functionality here (e.g. special
# vocab read)
EXTRA_ARGS="${@}"

SCRIPT=$(readlink -f "$0")
BASEDIR=$(readlink -f `dirname "$SCRIPT"`/../..)

for size in 100 200 300 400 500 600 700 800; do
    for window in 5 10; do
	for min_count in 20 100; do
	    eval $BASEDIR/tools/vsm/word2vec.sh "${CORPUS}" $EXTRA_ARGS \
		-size $size -window $window -mincount $min_count \
		-hs 1 -negative 0
	    eval $BASEDIR/tools/vsm/word2vec.sh "${CORPUS}" $EXTRA_ARGS \
		-size $size -window $window -mincount $min_count \
		-hs 0 -negative 5
	done
    done
done
