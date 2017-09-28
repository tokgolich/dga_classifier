import pandas as pd
from sklearn.svm import SVC
from sklearn.externals import joblib

raw = pd.read_csv('data/vectorized_feature_w_ranks_norm.txt')

X=raw.ix[:,'bi_rank':'vowel_ratio'].as_matrix()
Y=raw.ix[:,'class'].as_matrix()

clf = SVC(kernel='linear', probability=True, random_state=0)
clf.fit(X, Y)
joblib.dump(clf, 'data/dga_model.pkl')
#clf = joblib.load('data/dga_model.pkl')