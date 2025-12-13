"""
Image Feature Extraction theo ĐÚNG scikit-learn
- Patch Extraction (extract_patches_2d, PatchExtractor)
- Image-to-Graph Conversion (img_to_graph)
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image as sk_image
from sklearn.cluster import spectral_clustering

class ImageFeatureExtractor:
    def __init__(self, method='patches'):
        """
        method: 'patches' hoặc 'graph'
        """
        self.method = method
    
    def extract_patches(self, image_array, patch_size=(32, 32), max_patches=100):
        """
        Extract patches từ ảnh theo scikit-learn
        
        Parameters:
        -----------
        image_array : numpy array
            Ảnh input shape (H, W, C) hoặc (H, W)
        patch_size : tuple
            Kích thước mỗi patch (height, width)
        max_patches : int
            Số lượng patches tối đa (random sampling)
        
        Returns:
        --------
        patches : array shape (n_patches, patch_h, patch_w, C)
            Các patches đã extract
        patches_flat : array shape (n_patches, features)
            Patches đã flatten thành vectors
        """
        # Ensure 3D (H, W, C)
        if len(image_array.shape) == 2:
            # Grayscale -> RGB
            image_array = np.stack([image_array] * 3, axis=-1)
        
        # Extract patches using scikit-learn
        patches = sk_image.extract_patches_2d(
            image_array, 
            patch_size, 
            max_patches=max_patches,
            random_state=42
        )
        
        # Flatten each patch thành 1D vector
        patches_flat = patches.reshape(patches.shape[0], -1)
        
        return patches, patches_flat
    
    def image_to_graph(self, image_array, n_clusters=3):
        """
        Convert image to graph structure và thực hiện spectral clustering
        
        Parameters:
        -----------
        image_array : numpy array
            Ảnh input (H, W) hoặc (H, W, C)
        n_clusters : int
            Số clusters cho segmentation
        
        Returns:
        --------
        graph : sparse matrix
            Adjacency matrix của graph
        labels : array
            Cluster labels cho mỗi pixel
        segmented : array
            Ảnh đã segment
        """
        # Convert to grayscale if RGB
        if len(image_array.shape) == 3:
            gray = np.mean(image_array, axis=2).astype(np.uint8)
        else:
            gray = image_array
        
        # Resize nhỏ để xử lý nhanh hơn (graph-based clustering rất chậm với ảnh lớn)
        from PIL import Image as PILImage
        small_img = PILImage.fromarray(gray).resize((50, 50))
        small_array = np.array(small_img)
        
        # Convert image to graph using scikit-learn
        graph = sk_image.img_to_graph(small_array)
        
        # Spectral clustering trên graph
        labels = spectral_clustering(
            graph,
            n_clusters=n_clusters,
            eigen_solver='arpack',
            random_state=42
        )
        
        # Reshape labels về hình dạng ảnh
        segmented = labels.reshape(small_array.shape)
        
        return graph, labels, segmented, small_array
    
    def visualize_patches(self, patches, n_display=16):
        """
        Visualize extracted patches trong grid
        
        Parameters:
        -----------
        patches : array
            Patches đã extract
        n_display : int
            Số patches hiển thị
        """
        n_display = min(n_display, len(patches))
        n_cols = 4
        n_rows = (n_display + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        axes = axes.flatten()
        
        for i in range(n_display):
            # Normalize patch để hiển thị tốt hơn
            patch = patches[i]
            if patch.max() > 1:
                patch = patch / 255.0
            
            axes[i].imshow(patch)
            axes[i].set_title(f'Patch {i+1}\n{patch.shape[0]}×{patch.shape[1]}', fontsize=10)
            axes[i].axis('off')
        
        # Hide extra subplots
        for i in range(n_display, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def visualize_segmentation(self, original_image, segmented, small_image):
        """
        Visualize original image và segmented result
        
        Parameters:
        -----------
        original_image : array
            Ảnh gốc
        segmented : array
            Ảnh đã segment (cluster labels)
        small_image : array
            Ảnh đã resize nhỏ (cho graph)
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original full size
        if len(original_image.shape) == 3:
            axes[0].imshow(original_image)
        else:
            axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original Image\n(Full Size)', fontsize=12)
        axes[0].axis('off')
        
        # Resized for graph
        axes[1].imshow(small_image, cmap='gray')
        axes[1].set_title(f'Resized for Graph\n{small_image.shape[0]}×{small_image.shape[1]}', fontsize=12)
        axes[1].axis('off')
        
        # Segmented result
        axes[2].imshow(segmented, cmap='tab10')
        axes[2].set_title('Segmented Image\n(Spectral Clustering)', fontsize=12)
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def extract_features(self, image, **kwargs):
        """
        Main method để extract features theo method đã chọn
        
        Parameters:
        -----------
        image : PIL Image hoặc numpy array
            Ảnh input
        **kwargs : dict
            Các parameters cho từng method
        
        Returns:
        --------
        features : array
            Feature vector/array
        extras : dict
            Thông tin bổ sung (patches, graph, segmented, etc.)
        """
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        if self.method == 'patches':
            # Patch Extraction
            patches, patches_flat = self.extract_patches(
                image_array,
                patch_size=kwargs.get('patch_size', (32, 32)),
                max_patches=kwargs.get('max_patches', 100)
            )
            return patches_flat, {
                'patches': patches,
                'patch_size': patches.shape[1:3],
                'n_patches': len(patches)
            }
        
        elif self.method == 'graph':
            # Image-to-Graph Conversion
            graph, labels, segmented, small_image = self.image_to_graph(
                image_array,
                n_clusters=kwargs.get('n_clusters', 3)
            )
            return labels, {
                'graph': graph,
                'segmented': segmented,
                'small_image': small_image,
                'original': image_array,
                'n_clusters': kwargs.get('n_clusters', 3)
            }
        
        else:
            raise ValueError(f"Unknown method: {self.method}")


# Test function
def test_sklearn_image_features():
    """Test các functions"""
    print("=== Testing Sklearn Image Features ===\n")
    
    # Tạo ảnh test
    test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    # Test Patch Extraction
    print("1. Testing Patch Extraction:")
    extractor_patches = ImageFeatureExtractor(method='patches')
    features, extras = extractor_patches.extract_features(
        test_image,
        patch_size=(32, 32),
        max_patches=50
    )
    print(f"   ✓ Patches shape: {extras['patches'].shape}")
    print(f"   ✓ Flattened features shape: {features.shape}")
    print(f"   ✓ Number of patches: {extras['n_patches']}")
    print(f"   ✓ Each patch size: {extras['patch_size']}")
    
    print("\n2. Testing Image-to-Graph:")
    extractor_graph = ImageFeatureExtractor(method='graph')
    labels, extras_graph = extractor_graph.extract_features(
        test_image,
        n_clusters=3
    )
    print(f"   ✓ Graph shape: {extras_graph['graph'].shape}")
    print(f"   ✓ Number of nodes: {extras_graph['graph'].shape[0]}")
    print(f"   ✓ Labels shape: {labels.shape}")
    print(f"   ✓ Segmented shape: {extras_graph['segmented'].shape}")
    print(f"   ✓ Number of clusters: {extras_graph['n_clusters']}")
    
    # Count pixels per cluster
    unique, counts = np.unique(labels, return_counts=True)
    print(f"   ✓ Cluster distribution:")
    for cluster, count in zip(unique, counts):
        print(f"      Cluster {cluster}: {count} pixels")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_sklearn_image_features()
