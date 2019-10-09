
from sklearn import metrics
from sklearn.cluster import* #KMeans,AffinityPropagation,DBSCAN,MeanShift,SpectralClustering,AgglomerativeClustering,GaussianMixture
from time import time
import numpy as np
from sklearn import mixture
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
# #############################################################################
#DATA
digits = load_digits()
X = scale(digits.data)
labels = digits.target
n_digits = len(np.unique(digits.target))
#################################################
#OUT
print('%-30s\t%s\t%s\t\t%s\t\t%s' % ('', 'TIME', 'HOM', 'COM', 'NMI'))
def Output_result(labels_true,labels_pred,name):
    print('%-30s\t%.2fs\t%.3f\t%.3f\t%.3f' % (name,(time() - t0),
          metrics.homogeneity_score(labels_true,labels_pred),metrics.completeness_score(labels_true, labels_pred),metrics.normalized_mutual_info_score(labels_true, labels_pred,average_method='arithmetic')))
# #############################################################################
# Do the actual clustering
#km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,verbose=False)
km=KMeans(init='k-means++', n_clusters=n_digits, n_init=10,verbose=False)
t0 = time()
km.fit(X)
Output_result(labels,km.labels_,'KMeans')

af = AffinityPropagation(preference=-5000)
t0 = time()
af.fit(X)
Output_result(labels,af.labels_,'AffinityPropagation')

mf = MeanShift(bandwidth=6)
t0 = time()
mf.fit(X)
Output_result(labels,mf.labels_,'MeanShit')

sc = SpectralClustering(n_clusters=n_digits,affinity='nearest_neighbors')
t0 = time()
sc.fit(X)
Output_result(labels,sc.labels_,'SpectralClustering')

ac = AgglomerativeClustering(n_clusters=n_digits)
t0 = time()
ac.fit(X)
Output_result(labels,ac.labels_,'AgglomerativeClustering')

db = DBSCAN(eps=4, min_samples=5)
t0 = time()
db.fit(X)
Output_result(labels,db.labels_,'DBSCAN')

t0 = time()
gm = mixture.GaussianMixture(n_components=10).fit(X)
Output_result(labels,gm.predict(X),'GaussianMixture')

