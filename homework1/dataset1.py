from time import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn import metrics
from sklearn.cluster import *
from sklearn import mixture
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

np.random.seed(42)

digits = load_digits()
data = scale(digits.data)

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


print(82 * '_')
print('init\t\ttime\tNMI\tHomogeneity\tCompleteness')


def bench(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    if hasattr(estimator, 'labels_'):
        print('%-9s\t%.2fs\t%.3f\t%.3f\t\t%.3f'
          % (name, (time() - t0),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_)))
    else:
        print('%-9s\t%.2fs\t%.3f\t%.3f\t\t%.3f'
          % (name, (time() - t0),
             metrics.v_measure_score(estimator.predict(data), estimator.fit_predict(data)),
             metrics.homogeneity_score(estimator.predict(data), estimator.fit_predict(data)),
             metrics.completeness_score(estimator.predict(data), estimator.fit_predict(data))))

bench(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=data)
bench(AffinityPropagation(preference=-50),
     name="Affinity", data=data)

bench(MeanShift(bin_seeding=True),
              name="MeanShift", data=data)
bench(SpectralClustering(eigen_solver='arpack',affinity="nearest_neighbors",n_clusters=10),
             name="Spectral", data=data)

bench(AgglomerativeClustering(linkage='ward',n_clusters=10),
              name="Ward", data=data)
bench(DBSCAN(eps=3, min_samples=2),
              name="DBSCAN", data=data)
bench(mixture.GaussianMixture(n_components=10, covariance_type='full'),
     name="Gaussian", data=data)
print(82 * '_')

