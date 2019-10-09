from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
import numpy as np
from sklearn.decomposition import PCA
from time import time
# #############################################################################
digits = load_digits()
X = scale(digits.data)
labels_true = digits.target
# #############################################################################
# Compute DBSCAN
t0=time()
db = DBSCAN(eps=4, min_samples=5).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
print('name\t\ttime\thomo\tcompl\tNMI')
print('%-10s\t%.2fs\t%.3f\t%.3f\t%.3f'
          % ('DBSCAN', (time() - t0),
             metrics.homogeneity_score(labels_true, labels),
             metrics.completeness_score(labels_true, labels),
             metrics.normalized_mutual_info_score(labels_true, labels,average_method='arithmetic')))
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
# #############################################################################
# Plot result
import matplotlib.pyplot as plt
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
    class_member_mask = (labels == k)
    X = PCA(n_components=2).fit_transform(X)
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=7)
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=3)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
