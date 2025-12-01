"""
Module xử lý Image Feature Extraction
Methods: Color Histogram, HOG (Histogram of Oriented Gradients)
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ImageFeatureExtractor:
    def __init__(self, method='histogram'):
        """
        method: 'histogram', 'hog', 'edges'
        """
        self.method = method
    
    def extract_color_histogram(self, image, bins=32):
        """
        Trích xuất color histogram
        Input: Image (numpy array hoặc PIL Image)
        Output: Feature vector
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Chuyển sang RGB nếu cần
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Tính histogram cho mỗi channel
        features = []
        for i in range(3):  # R, G, B
            hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
            hist = hist.flatten()
            features.extend(hist)
        
        # Normalize
        features = np.array(features)
        features = features / (features.sum() + 1e-7)
        
        return features
    
    def extract_hog_features(self, image):
        """
        Trích xuất HOG (Histogram of Oriented Gradients)
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Chuyển sang grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Resize về kích thước chuẩn
        gray = cv2.resize(gray, (128, 128))
        
        # Tính HOG
        from skimage.feature import hog
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=False)
        
        return features
    
    def extract_edge_features(self, image):
        """
        Trích xuất edge features sử dụng Canny
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Chuyển sang grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect edges
        edges = cv2.Canny(gray, 100, 200)
        
        # Resize và flatten
        edges = cv2.resize(edges, (32, 32))
        features = edges.flatten() / 255.0
        
        return features, edges
    
    def extract_features(self, image):
        """
        Extract features theo method được chọn
        """
        if self.method == 'histogram':
            return self.extract_color_histogram(image)
        elif self.method == 'hog':
            return self.extract_hog_features(image)
        elif self.method == 'edges':
            features, _ = self.extract_edge_features(image)
            return features
        
    def visualize_features(self, image):
        """
        Visualize extracted features
        """
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        if self.method == 'histogram':
            # Vẽ color histogram
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Original image
            axes[0].imshow(image_array)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Histograms
            colors = ('r', 'g', 'b')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([image_array], [i], None, [32], [0, 256])
                axes[1].plot(hist, color=color, label=color.upper())
            
            axes[1].set_title('Color Histograms')
            axes[1].set_xlabel('Bins')
            axes[1].set_ylabel('Frequency')
            axes[1].legend()
            
            return fig
        
        elif self.method == 'edges':
            features, edges = self.extract_edge_features(image_array)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].imshow(image_array)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(edges, cmap='gray')
            axes[1].set_title('Edge Detection')
            axes[1].axis('off')
            
            return fig

# Test function
def test_image_features():
    # Tạo ảnh test đơn giản
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    print("=== Color Histogram ===")
    extractor_hist = ImageFeatureExtractor(method='histogram')
    features_hist = extractor_hist.extract_features(image)
    print(f"Shape: {features_hist.shape}")
    print(f"First 10 values: {features_hist[:10]}")
    
    print("\n=== HOG Features ===")
    extractor_hog = ImageFeatureExtractor(method='hog')
    features_hog = extractor_hog.extract_features(image)
    print(f"Shape: {features_hog.shape}")
    
    print("\n=== Edge Features ===")
    extractor_edge = ImageFeatureExtractor(method='edges')
    features_edge = extractor_edge.extract_features(image)
    print(f"Shape: {features_edge.shape}")

if __name__ == "__main__":
    test_image_features()
