# just a test for matplotlib

# import pylab

# x = randn(10000)
# hist(x, 100)

import pylab
import numpy as np
import pickle

enData = pickle.load(open('data/tsne-en.p','r'))
enLabels = pickle.load(open('data/vocab-en.p','r'))
esData = pickle.load(open('data/tsne-es.p','r'))
esLabels = pickle.load(open('data/vocab-es.p','r'))
# enVsm = pickle.load(open('data/vsm-en.p','r'))
# esVsm = pickle.load(open('data/vsm-es.p','r'))
# tn = pickle.load(open('translation-network.p','r'))

#pylab.scatter(myData[:,0],myData[:,1],0)
def annotator(data, labels, lang):
    labelStyle = 'normal' if lang == 'en' else 'italic'
    color = 'blue' if lang == 'en' else 'red'
    for label, x, y in zip(labels, data[:, 0], data[:, 1]):
        if label in ['man','hombre','woman','mujer']:
            pylab.scatter([x],[y],20,[color])
            pylab.annotate(label, xy = (x, y), style = labelStyle) 
    
annotator(enData,enLabels,'en')
annotator(esData,esLabels,'es')
pylab.show()

# annotator(data,labels)
# plt.show()

# code copied and modified from http://stackoverflow.com/questions/5147112/matplotlib-how-to-put-individual-tags-for-a-scatter-plot
