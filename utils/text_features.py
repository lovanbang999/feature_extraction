"""
Module xử lý Text Feature Extraction
Giải thích: Chuyển đổi text thành numerical vectors
Methods: CountVectorizer, TfidfVectorizer
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class TextFeatureExtractor:
    def __init__(self, method='tfidf', max_features=50):
        self.method = method
        if method == 'count':
            self.vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
        elif method == 'hashing' : 
            self.vectorizer = HashingVectorizer(n_features=max_features, alternate_sign=False)
        else:
            self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    
    def extract_features(self, texts):
    
        try:
            if self.method == 'hashing':
                features = self.vectorizer.transform(texts).toarray()
                feature_names = [f'feature_{i}' for i in range(features.shape[1])]
            else:
                features = self.vectorizer.fit_transform(texts).toarray()
                feature_names = self.vectorizer.get_feature_names_out()
            
            return features, feature_names
        except Exception as e:
            return None, str(e)
    
    def analyze_text(self, texts):
   
        features, feature_names = self.extract_features(texts)
        
        word_scores = features.sum(axis=0)
        
        # Top words
        top_indices = np.argsort(word_scores)[-15:][::-1]
        top_words = [(feature_names[i], word_scores[i]) for i in top_indices]
        
        return features, feature_names, top_words
    
    def visualize_features(self, texts, feature_names, features):
        
        import plotly.graph_objects as go
        
        # Lấy top 15 words
        word_scores = features.sum(axis=0)
        top_indices = np.argsort(word_scores)[-10:][::-1]
        
        fig = go.Figure(data=[
            go.Bar(
                x=[feature_names[i] for i in top_indices],
                y=[word_scores[i] for i in top_indices],
                marker_color='lightblue'
            )
        ])
        
        fig.update_layout(
            title=f'Top 15 Words ({self.method.upper()})',
            xaxis_title='Words',
            yaxis_title='Score',
            height=400
        )
        
        return fig

def test_text_features():
    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Data science combines statistics and programming",
        "Python is the most popular language for machine learning",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing deals with text and speech"
    ]
    
    # Test TF-IDF
    print("=== TF-IDF Features ===")
    extractor_tfidf = TextFeatureExtractor(method='tfidf', max_features=20)
    features_tfidf, names_tfidf = extractor_tfidf.extract_features(texts)
    print("Feature names (tfidf):")
    print(names_tfidf)
    print("\nFeature matrix (tfidf):")
    print(features_tfidf)
    
    # Test Count
    print("\n=== Count Features ===")
    extractor_count = TextFeatureExtractor(method='count', max_features=20)
    features_count, names_count = extractor_count.extract_features(texts)
    print("Feature names (count):")
    print(names_count)
    print("\nFeature matrix (count):")
    print(features_count)

    # Test Hashing
    print("\n=== Hashing Features ===")
    extractor_hash = TextFeatureExtractor(method='hashing', max_features=20)
    features_hash, names_hash = extractor_hash.extract_features(texts)
    print("Pseudo feature names (hashing):")
    print(names_hash)
    print("\nFeature matrix (hashing):")
    print(features_hash)

if __name__ == "__main__":
    test_text_features()
