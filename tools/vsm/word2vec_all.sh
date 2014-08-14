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

MIN_COUNT=40

for window in 5 10; do
    for size in 100 200 300 400 500 600 700 800; do
	    echo "${BASEDIR}/tools/vsm/word2vec.sh ${CORPUS} ${EXTRA_ARGS}"
	    eval $BASEDIR/tools/vsm/word2vec.sh "${CORPUS}" $EXTRA_ARGS \
		-size $size -window $window -mincount $MIN_COUNT \
		-hs 0 -negative 5 -sample 1e-5
    done
done
