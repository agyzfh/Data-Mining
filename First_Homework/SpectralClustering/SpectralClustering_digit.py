from sklearn.cluster import SpectralClustering
from sklearn import metrics
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from time import time
# #############################################################################
#sample data
digits = load_digits()
data = scale(digits.data)
labels_true = digits.target
n_digits = len(np.unique(digits.target))
# Compute
t0=time()
sc = SpectralClustering(n_clusters=n_digits,affinity='nearest_neighbors').fit(data)
labels = sc.labels_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print('name\t\t\t\ttime\thomo\tcompl\tNMI')
print('%-15s\t%.2fs\t%.3f\t%.3f\t%.3f'
          % ('SpectralClustering', (time() - t0),
             metrics.homogeneity_score(labels_true, labels),
             metrics.completeness_score(labels_true, labels),
             metrics.normalized_mutual_info_score(labels_true, labels,average_method='arithmetic')))
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle
plt.close('all')
plt.figure(1)
plt.clf()
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    data = PCA(n_components=2).fit_transform(data)
    plt.plot(data[class_members, 0], data[class_members, 1], col + '.')
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
