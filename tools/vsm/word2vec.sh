#!/bin/zsh

NAME=$0

usage() {
    echo "Usage: $NAME <corpus> [word2vec-args]
	
	Where <corpus> is a file in data/corpus.

	The corpus name and the parameters will be used to construct a
	VSM filename." 1>&2

    exit 1
}

if [ $# -lt 1 ]; then
    usage; exit 1
fi

# Get path of project directory
SCRIPT=$(readlink -f "$0")
BASEDIR=$(dirname `dirname "$SCRIPT"`)

CORPUS_NAME=$1
CORPUS_PATH=`readlink -f "$BASEDIR/data/corpus/$CORPUS_NAME"`
if [ ! -f $CORPUS_PATH ]; then
    echo "Corpus file $CORPUS_PATH does not exist." 1>&2
    exit 1
fi

# Shift off the language argument -- the rest goes to word2vec
shift

WORD2VEC_ARGS="$@"

# Save the training-specific args (i.e., not the mundane ones) so that
# we can create a useful model filename later
TRAINING_ARGS=$WORD2VEC_ARGS

# Build an output filename
TRAIN_ARGS_SQUISHED=${TRAINING_ARGS//[[:blank:]]/}
OUTPUT_NAME="$CORPUS_NAME$TRAIN_ARGS_SQUISHED"
OUTPUT_PATH="$BASEDIR/data/vsm/$OUTPUT_NAME"

if [ -f $OUTPUT_PATH ]; then
    echo -n "Output VSM path exists: $OUTPUT_PATH
	Overwrite? [y/N] "
    read REPLY

    if ! [[ $REPLY =~ ^[Yy]$ ]]; then
	echo "OK, exiting."
	exit 1
    fi
fi

# OK, start appending boring arguments
WORD2VEC_ARGS="-train $CORPUS_PATH -output $OUTPUT_PATH $WORD2VEC_ARGS"

# Have we learned a vocab for this corpus yet?
VOCAB_PATH=$BASEDIR/data/vocab/$CORPUS_NAME
if [ -f $VOCAB_PATH ]; then
    WORD2VEC_ARGS="-read-vocab $VOCAB_PATH $WORD2VEC_ARGS"
else
    WORD2VEC_ARGS="-save-vocab $VOCAB_PATH $WORD2VEC_ARGS"
fi

# Multithreading
NTHREADS=$((`nproc` > 16 ? 16 : `nproc`))
WORD2VEC_ARGS="-threads `nproc` $WORD2VEC_ARGS"

# Use more condensed binary format
WORD2VEC_ARGS="-binary 1 $WORD2VEC_ARGS"

# Time to go!
echo "Invoking word2vec with arguments:

    $WORD2VEC_ARGS
    
Writing output to $BASEDIR/log/$OUTPUT_NAME.{err,out}"

WORD2VEC=/u/nlp/packages/word2vec/word2vec
eval $WORD2VEC $WORD2VEC_ARGS > "$BASEDIR/log/$OUTPUT_NAME.out" 2> "$BASEDIR/log/$OUTPUT_NAME.err"
