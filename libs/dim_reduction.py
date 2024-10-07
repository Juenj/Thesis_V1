from abc import ABC, abstractmethod
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Abstract base class for dimensionality reduction objects
class DimRedObject(ABC):
  
    def __init__(self, data: np.ndarray):
        """
        Initialize the DimRedObject with input data.
        
        Parameters:
        - data: np.ndarray - The data to be used for dimensionality reduction.
        """
        self.data = data
    

    @abstractmethod
    def transform(self, new_data: np.ndarray):
        """
        Abstract method for transforming new data.
        Must be implemented by subclasses.
        
        Parameters:
        - new_data: np.ndarray - The new data to be transformed.
        """
        pass

# PCAObject class that implements the DimRedObject interface using PCA
class PCAObject(DimRedObject):
    
    def __init__(self, data: np.ndarray, n_components=10):
        """
        Initialize the PCAObject with input data and number of components. Automatically scales data
        
        Parameters:
        - data: np.ndarray - The data to be reduced using PCA.
        - n_components: int - Number of principal components to retain (default is 10).
        """
        super().__init__(data)

        self.scaler = StandardScaler().fit(data)
        data = self.scaler.transform(data)
        self.pca = PCA(n_components).fit(data)  # Initialize the PCA model with n components
        #self.pca.fit(self.data)  # Fit the PCA model to the input data
    
    def transform(self, new_data: np.ndarray):
        """
        Transform the new data using the fitted PCA model.
        
        Parameters:
        - new_data: np.ndarray - The data to transform.
        
        Returns:
        - np.ndarray - The transformed data in the PCA-reduced space.
        """
        return self.pca.transform(self.scaler.transform(new_data))
    
    def get_components(self):
        """
        Get the principal components after fitting the PCA model.
        
        Returns:
        - np.ndarray - The principal components of the PCA model.
        """
        return self.pca.components_
