import numpy as np
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans

class ClusteringObject(ABC):
  
    def __init__(self, data : np.ndarray, n_clusters):
        self.data = data
        self.n_clusters = n_clusters

    @abstractmethod
    def cluster(self):
        pass


class KMeansObject(ClusteringObject):
    
    def __init__(self, data: np.ndarray):
        super().__init__(data)
        self.KMeans = KMeans(self.n_clusters)
        self.KMeans.fit(self.data)
    
    def cluster(self):
        return self.KMeans.labels_

