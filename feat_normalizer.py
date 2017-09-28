'''
this script reads in the feature table before vectorizing, and normalize all numerical features from 0 to 1
'''
import pandas as pd
import numpy as np
black_list = ['ip','class','tld']

feat_table = pd.read_csv('data/features.txt',delimiter='\t')
fw = open('data/features_norm_para.txt','w')

header = list(feat_table.columns)
feat_matrix = pd.DataFrame()
for i in header:
    if i in black_list:
        feat_matrix[i]=feat_table.ix[:,i]
    else:
        line = feat_table.ix[:,i].as_matrix()
        mean_ = line.mean()
        max_ = line.max()
        min_ = line.min()
        feat_matrix[i]=(line-mean_)/(max_-min_)
        fw.write('%s\t%f\t%f\t%f\n'%(i,mean_,max_,min_))
    print 'converted %s'%i

feat_matrix.to_csv('data/features_norm.txt')
fw.close()
