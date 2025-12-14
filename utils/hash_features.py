from sklearn.feature_extraction import FeatureHasher,DictVectorizer
import pandas as pd
import numpy as np

class HashFeatureExtractor:
    # khởi tạo
    def __init__(self, n_features=10):
        self.n_features = n_features
        self.hasher = FeatureHasher(n_features=n_features, input_type='dict')

    # hàm trích xuất đặc trưng  
    def extract_features(self, data_dict_list):
        """
        Input: List of dictionaries
        Output: Hashed feature matrix
        """
        try:
            # Transform data
            # features = self.hasher.transform(data_dict_list).toarray()
            hashed_matrix = self.hasher.transform(data_dict_list)
            features = hashed_matrix.toarray()
            return features
        except Exception as e:
            return None
    
    #hàm so sánh feature hashing với dictVectorizer
    def compare_with_dict_vectorizer(self, data_dict_list):       
        # DictVectorizer
        dict_vec = DictVectorizer(sparse=False)
        dict_features = dict_vec.fit_transform(data_dict_list)
        
        # Feature Hashing
        hash_features = self.extract_features(data_dict_list)
        comparison = {
            'DictVectorizer': {
                'n_features': dict_features.shape[1],
                'shape': dict_features.shape,
                'memory_size': dict_features.nbytes
            },
            'FeatureHasher': {
                'n_features': self.n_features,
                'shape': hash_features.shape,
                'memory_size': hash_features.nbytes
            }
        }
        return comparison, dict_features, hash_features
    
# Test function
def test_hash_features():
    data = [
        {'country': 'Vietnam', 'city': 'Hanoi', 'age': 25},
        {'country': 'Vietnam', 'city': 'HCM', 'age': 30},
        {'country': 'Thailand', 'city': 'Bangkok', 'age': 28}
    ]
    extractor = HashFeatureExtractor(n_features=8)
    features = extractor.extract_features(data)

    print("Original data:")
    print(data)
    print("\nHashed features (8 dimensions):")
    print(features)
    
    # So sánh
    comparison, _, _ = extractor.compare_with_dict_vectorizer(data)
    print("\nComparison:")
    print(f"DictVectorizer: {comparison['DictVectorizer']}")
    print(f"FeatureHasher: {comparison['FeatureHasher']}")  

if __name__ == "__main__":
    test_hash_features()
