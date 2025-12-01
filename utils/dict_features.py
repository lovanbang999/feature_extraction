"""
Module xử lý feature extraction từ dictionary data
Giải thích: Chuyển đổi dữ liệu dạng dict (categorical) thành vector số
"""

from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np

class DictFeatureExtractor:
    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=False)
        
    def extract_features(self, data_dict_list):
        """
        Input: List of dictionaries
        Output: Feature matrix (numpy array)
        
        Ví dụ:
        Input: [{'city': 'Hanoi', 'age': 25}, {'city': 'HCM', 'age': 30}]
        Output: [[25, 0, 1], [30, 1, 0]]  # age, city=HCM, city=Hanoi
        """
        try:
            # Fit và transform data
            features = self.vectorizer.fit_transform(data_dict_list)
            feature_names = self.vectorizer.get_feature_names_out()
            
            return features, feature_names
        except Exception as e:
            return None, str(e)
    
    def demo_with_titanic(self, df, n_samples=5):
        """
        Demo với Titanic dataset
        """
        # Chọn các cột categorical
        df_sample = df[['Pclass', 'Sex', 'Embarked']].head(n_samples).fillna('Unknown')
        
        # Chuyển thành list of dicts
        dict_data = df_sample.to_dict('records')
        
        # Extract features
        features, feature_names = self.extract_features(dict_data)
        
        return dict_data, features, feature_names

# Test function
def test_dict_features():
    # Ví dụ đơn giản
    data = [
        {'city': 'Hanoi', 'gender': 'Male', 'age': 25},
        {'city': 'HCM', 'gender': 'Female', 'age': 30},
        {'city': 'Hanoi', 'gender': 'Male', 'age': 22}
    ]
    
    extractor = DictFeatureExtractor()
    features, names = extractor.extract_features(data)
    
    print("Original data:")
    print(data)
    print("\nFeature names:")
    print(names)
    print("\nFeature matrix:")
    print(features)

if __name__ == "__main__":
    test_dict_features()
