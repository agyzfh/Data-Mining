from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
import numpy as np
from sklearn.decomposition import PCA
from time import time
# #############################################################################
#sample data
digits = load_digits()
data = scale(digits.data)
n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels_true = digits.target

# Compute Affinity Propagation
t0 = time()
af = AffinityPropagation(preference=-5000).fit(data)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
n_clusters_ = len(cluster_centers_indices)
print('name\t\t\t\ttime\thomo\tcompl\tNMI')
print('%-15s\t%.2fs\t%.3f\t%.3f\t%.3f'
          % ('AffinityPropagation', (time() - t0),
             metrics.homogeneity_score(labels_true, labels),
             metrics.completeness_score(labels_true, labels),
             metrics.normalized_mutual_info_score(labels_true, labels,average_method='arithmetic'),))
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle
plt.close('all')
plt.figure(1)
plt.clf()
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = data[cluster_centers_indices[k]]
    data = PCA(n_components=2).fit_transform(data)
    plt.plot(data[class_members, 0], data[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=14)
    for x in data[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
