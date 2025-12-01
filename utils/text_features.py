"""
Module xử lý Text Feature Extraction
Giải thích: Chuyển đổi text thành numerical vectors
Methods: CountVectorizer, TfidfVectorizer
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class TextFeatureExtractor:
    def __init__(self, method='tfidf', max_features=50):
        """
        method: 'count' hoặc 'tfidf'
        max_features: Số lượng features tối đa
        """
        self.method = method
        if method == 'count':
            self.vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
        else:
            self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    
    def extract_features(self, texts):
        """
        Input: List of text strings
        Output: Feature matrix
        """
        try:
            features = self.vectorizer.fit_transform(texts).toarray()
            feature_names = self.vectorizer.get_feature_names_out()
            
            return features, feature_names
        except Exception as e:
            return None, str(e)
    
    def analyze_text(self, texts):
        """
        Phân tích chi tiết text
        """
        features, feature_names = self.extract_features(texts)
        
        # Tính tổng score cho mỗi từ
        word_scores = features.sum(axis=0)
        
        # Top words
        top_indices = np.argsort(word_scores)[-10:][::-1]
        top_words = [(feature_names[i], word_scores[i]) for i in top_indices]
        
        return features, feature_names, top_words
    
    def visualize_features(self, texts, feature_names, features):
        """
        Visualize text features
        """
        import plotly.graph_objects as go
        
        # Lấy top 15 words
        word_scores = features.sum(axis=0)
        top_indices = np.argsort(word_scores)[-15:][::-1]
        
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

# Test function
def test_text_features():
    texts = [
        "I love machine learning and data science",
        "Python is great for data analysis",
        "Machine learning is a subset of artificial intelligence",
        "Data science requires statistics and programming skills"
    ]
    
    # Test TF-IDF
    print("=== TF-IDF Features ===")
    extractor_tfidf = TextFeatureExtractor(method='tfidf', max_features=20)
    features, names = extractor_tfidf.extract_features(texts)
    
    print("Feature names:")
    print(names)
    print("\nFeature matrix:")
    print(features)
    
    # Test Count
    print("\n=== Count Features ===")
    extractor_count = TextFeatureExtractor(method='count', max_features=20)
    features_count, names_count = extractor_count.extract_features(texts)
    print("Feature matrix:")
    print(features_count)

if __name__ == "__main__":
    test_text_features()
