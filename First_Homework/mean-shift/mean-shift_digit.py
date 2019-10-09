from sklearn.cluster import MeanShift,estimate_bandwidth
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
# Compute
#bandwidth = estimate_bandwidth(data, quantile=0.6, n_samples=500)
t0=time()
ms = MeanShift(bandwidth=6)
ms.fit(data)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print('name\t\ttime\thomo\tcompl\tNMI')
print('%-10s\t%.2fs\t%.3f\t%.3f\t%.3f'
          % ('MeanShift', (time() - t0),
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
    cluster_center = cluster_centers[k]
    data = PCA(n_components=2).fit_transform(data)
    plt.plot(data[class_members, 0], data[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
