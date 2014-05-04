import sys
import codecs

def main(vsm1_path, vsm2_path, out_path):
    """Merges two VSMs, e.g. the English VSM translated into the Spanish
    space with the original Spanish VSM (only works for Word2Vec
    format)"""
    
    with codecs.open(vsm1_path,'r',encoding='utf-8') as f, \
         codecs.open(vsm2_path,'r',encoding='utf-8') as g, \
         codecs.open(out_path,'w',encoding='utf-8') as h:

        cap_f = f.readline().split() # first line: nVocab, nDim
        cap_g = g.readline().split()

        nVocab = int(cap_f[0])+int(cap_g[0])
        assert cap_f[1] == cap_g[1]
        nDim = cap_f[1]
        h.write(str(nVocab) + ' ' + str(nDim) + '\n')

        # now just copy all the words from both spaces
        for line in f:
            h.write(line)
        for line in g:
            h.write(line)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])

#python merge_vsm.py data/vsm-trans.w2v data/vsm-es.w2v data/vsm-tot.w2v
