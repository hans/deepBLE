To learn a model:

    python 1-fetch_bilingual.py > data/bilingual-sample.tsv
    python 2-build_corpus.py data/bilingual-sample.tsv 0 en > data/corpus-en.txt
    python 2-build_corpus.py data/bilingual-sample.tsv 1 es > data/corpus-es.txt
    python 3-build_vsm.py data/corpus-en.txt data/vsm-en.bin
    python 3-build_vsm.py data/corpus-es.txt data/vsm-es.bin
    python 4-learn_translation_model.py data/vsm-en.bin data/vsm-es.bin data/seed-set.tsv

Load the learned model (saved in a file `translation-network`) with `pickle`.
