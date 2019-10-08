from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn import mixture
from time import time
import numpy as np

# #############################################################################
# Load some categories from the training set
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
dataset = fetch_20newsgroups(subset='all', categories=categories,shuffle=True, random_state=42)
print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
labels = dataset.target
true_k = np.unique(labels).shape[0]

vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000, min_df=2, stop_words='english', use_idf=True)
X = vectorizer.fit_transform(dataset.data)
# Vectorizer results are normalized, which makes KMeans behave as
# spherical k-means for better results. Since LSA/SVD results are
# not normalized, we have to redo the normalization.
svd = TruncatedSVD(3)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)
# #############################################################################
# Do the actual clustering
t0 = time()
gm = mixture.GaussianMixture(n_components=true_k).fit_predict(X)
print("done in %0.3fs" % (time() - t0))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels,gm))
print("Completeness: %0.3f" % metrics.completeness_score(labels,gm))
print("NMI: %0.3f" % metrics.normalized_mutual_info_score(labels,gm))


