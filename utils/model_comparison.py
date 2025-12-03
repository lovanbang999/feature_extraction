"""
So sánh hiệu suất các phương pháp feature extraction với machine learning
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import numpy as np

class FeatureExtractionComparison:
    def __init__(self):
        self.results = {}
    
    def compare_dict_vs_hash(self, data_dict_list, labels):
        """
        So sánh DictVectorizer vs FeatureHasher với ML model
        """
        from sklearn.feature_extraction import DictVectorizer, FeatureHasher
        
        # Split data
        train_data, test_data, y_train, y_test = train_test_split(
            data_dict_list, labels, test_size=0.3, random_state=42
        )
        
        results = {}
        
        # Test DictVectorizer
        print("Testing DictVectorizer...")
        dict_vec = DictVectorizer(sparse=False)
        
        start = time.time()
        X_train_dict = dict_vec.fit_transform(train_data)
        X_test_dict = dict_vec.transform(test_data)
        transform_time_dict = time.time() - start
        
        model = LogisticRegression(max_iter=1000)
        start = time.time()
        model.fit(X_train_dict, y_train)
        train_time_dict = time.time() - start
        
        y_pred = model.predict(X_test_dict)
        accuracy_dict = accuracy_score(y_test, y_pred)
        
        results['DictVectorizer'] = {
            'n_features': X_train_dict.shape[1],
            'transform_time': transform_time_dict,
            'train_time': train_time_dict,
            'accuracy': accuracy_dict,
            'memory_mb': X_train_dict.nbytes / (1024 * 1024)
        }
        
        # Test FeatureHasher
        print("Testing FeatureHasher...")
        hash_vec = FeatureHasher(n_features=50, input_type='dict')
        
        start = time.time()
        X_train_hash = hash_vec.transform(train_data).toarray()
        X_test_hash = hash_vec.transform(test_data).toarray()
        transform_time_hash = time.time() - start
        
        model = LogisticRegression(max_iter=1000)
        start = time.time()
        model.fit(X_train_hash, y_train)
        train_time_hash = time.time() - start
        
        y_pred = model.predict(X_test_hash)
        accuracy_hash = accuracy_score(y_test, y_pred)
        
        results['FeatureHasher'] = {
            'n_features': X_train_hash.shape[1],
            'transform_time': transform_time_hash,
            'train_time': train_time_hash,
            'accuracy': accuracy_hash,
            'memory_mb': X_train_hash.nbytes / (1024 * 1024)
        }
        
        return results
    
    def compare_text_methods(self, texts, labels):
        """
        So sánh Count vs TF-IDF
        """
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        
        train_texts, test_texts, y_train, y_test = train_test_split(
            texts, labels, test_size=0.3, random_state=42
        )
        
        results = {}
        
        # Test CountVectorizer
        count_vec = CountVectorizer(max_features=100, stop_words='english')
        start = time.time()
        X_train_count = count_vec.fit_transform(train_texts).toarray()
        X_test_count = count_vec.transform(test_texts).toarray()
        transform_time = time.time() - start
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_count, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test_count))
        
        results['CountVectorizer'] = {
            'transform_time': transform_time,
            'accuracy': accuracy,
            'n_features': X_train_count.shape[1]
        }
        
        # Test TfidfVectorizer
        tfidf_vec = TfidfVectorizer(max_features=100, stop_words='english')
        start = time.time()
        X_train_tfidf = tfidf_vec.fit_transform(train_texts).toarray()
        X_test_tfidf = tfidf_vec.transform(test_texts).toarray()
        transform_time = time.time() - start
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_tfidf, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test_tfidf))
        
        results['TfidfVectorizer'] = {
            'transform_time': transform_time,
            'accuracy': accuracy,
            'n_features': X_train_tfidf.shape[1]
        }
        
        return results
