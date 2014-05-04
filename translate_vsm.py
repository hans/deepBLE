import sys
import pickle
import codecs

import numpy as np

from gensim.models import Word2Vec

# run one VSM through the pyBrain model into the other
def main(tn_path, inVsm_path, outVsm_path):
    tn = pickle.load(open(tn_path,'r'))
    #inVsm = Word2Vec.load_word2vec_format(inVsm_path)
    #print 'part ii done'
    #inVsm = pickle.load(open(inVsm_path,'r'))

    # with codecs.open(inVsm_path, 'r') as f:
        
    def stringify(myArray):
        myString = ''
        for number in myArray:
            myString += ' ' + str(number)
        return myString

    with codecs.open(inVsm_path,'r',encoding='utf-8') as f:
        with codecs.open(outVsm_path,'w',encoding='utf-8') as g:
            g.write(f.readline())
            for line in f:
                word, arr0 = line.split()[0], line.split()[1:]
                arr1 = tn.activate(np.array(arr0).astype(np.float64))
                g.write('*' + word + stringify(arr1) + '\n')

    # with codecs.open(outVsm_path,'w',encoding='utf-8') as f:
    #     f.write(str(len(inVsm.vocab)) + ' 100\n')
    #     for array,word in zip(inVsm.syn0,inVsm.vocab):
    #         f.write('*'+word)
    #         for number in tn.activate(array):
    #             f.write(' '+str(number))
    #         f.write('\n')

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2],sys.argv[3])

#python translate_vsm.py translation-network.p data/vsm-en.w2v data/vsm-trans.w2v
