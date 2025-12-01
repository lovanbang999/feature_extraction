"""
WEB APP FEATURE EXTRACTION DEMO
Streamlit application tích hợp 4 phương pháp feature extraction
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from dict_features import DictFeatureExtractor
from hash_features import HashFeatureExtractor
from text_features import TextFeatureExtractor
from image_features import ImageFeatureExtractor

# Page config
st.set_page_config(
    page_title="Feature Extraction Demo",
    page_icon="🚀",
    layout="wide"
)

# Title
st.title("🚀 Feature Extraction Demo")
st.markdown("**Nhóm X - Môn Khai Phá Dữ Liệu**")
st.markdown("---")

# Sidebar
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio(
    "Chọn phương pháp:",
    ["🏠 Tổng quan", "📊 Dict Features", "# Feature Hashing", "📝 Text Features", "🖼️ Image Features", "🔬 So sánh"]
)

# ==================== TRANG TỔNG QUAN ====================
if page == "🏠 Tổng quan":
    st.header("Giới thiệu Feature Extraction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📖 Feature Extraction là gì?")
        st.write("""
        Feature Extraction (Trích xuất đặc trưng) là quá trình chuyển đổi dữ liệu thô 
        (raw data) thành dạng số (numerical) mà các thuật toán machine learning có thể hiểu được.
        
        **Tại sao cần Feature Extraction?**
        - Machine learning chỉ làm việc với số
        - Giảm chiều dữ liệu (dimensionality reduction)
        - Giữ lại thông tin quan trọng
        - Cải thiện hiệu suất model
        """)
    
    with col2:
        st.subheader("🎯 4 Phương pháp trong Demo")
        st.write("""
        1. **Dict Features**: Chuyển dictionary → vector
        2. **Feature Hashing**: Hash trick cho high-dimensional data
        3. **Text Features**: Chuyển text → vector (TF-IDF, Count)
        4. **Image Features**: Trích xuất đặc trưng từ ảnh
        """)
    
    st.markdown("---")
    st.subheader("📊 Quy trình chung")
    st.image("https://miro.medium.com/max/1400/1*VQvV5kVXZvHlmBDxWcZvNg.png", 
             caption="Feature Extraction Pipeline", use_column_width=True)
    
    st.info("👈 Chọn phương pháp ở sidebar để bắt đầu demo!")

# ==================== DICT FEATURES ====================
elif page == "📊 Dict Features":
    st.header("7.2.1. Loading Features from Dicts")
    
    st.subheader("📚 Lý thuyết")
    st.write("""
    **DictVectorizer** chuyển đổi dữ liệu dạng dictionary thành feature vectors.
    
    **Ví dụ:**
```python
    Input:  [{'city': 'Hanoi', 'age': 25}, {'city': 'HCM', 'age': 30}]
    Output: [[25, 0, 1], [30, 1, 0]]  # [age, city=HCM, city=Hanoi]
```
    
    **Ứng dụng:** Dữ liệu categorical như thông tin khách hàng, sản phẩm, v.v.
    """)
    
    st.markdown("---")
    st.subheader("🎮 Demo Interactive")
    
    # Chọn mode
    demo_mode = st.radio("Chọn mode:", ["Ví dụ đơn giản", "Titanic Dataset"])
    
    if demo_mode == "Ví dụ đơn giản":
        st.write("**Nhập dữ liệu của bạn:**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            name1 = st.text_input("Tên 1:", "An")
            city1 = st.selectbox("Thành phố 1:", ["Hanoi", "HCM", "Danang"], key="city1")
            age1 = st.number_input("Tuổi 1:", 18, 100, 25)
        
        with col2:
            name2 = st.text_input("Tên 2:", "Binh")
            city2 = st.selectbox("Thành phố 2:", ["Hanoi", "HCM", "Danang"], key="city2")
            age2 = st.number_input("Tuổi 2:", 18, 100, 30)
        
        with col3:
            name3 = st.text_input("Tên 3:", "Chi")
            city3 = st.selectbox("Thành phố 3:", ["Hanoi", "HCM", "Danang"], key="city3")
            age3 = st.number_input("Tuổi 3:", 18, 100, 22)
        
        if st.button("🚀 Extract Features"):
            data = [
                {'name': name1, 'city': city1, 'age': age1},
                {'name': name2, 'city': city2, 'age': age2},
                {'name': name3, 'city': city3, 'age': age3}
            ]
            
            extractor = DictFeatureExtractor()
            features, feature_names = extractor.extract_features(data)
            
            st.success("✅ Extraction hoàn tất!")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.write("**Input Data:**")
                st.json(data)
            
            with col_b:
                st.write("**Feature Names:**")
                st.write(list(feature_names))
            
            st.write("**Feature Matrix:**")
            df_features = pd.DataFrame(features, columns=feature_names)
            st.dataframe(df_features, use_container_width=True)
            
            st.write(f"📏 **Shape:** {features.shape} (3 samples × {features.shape[1]} features)")
    
    else:  # Titanic Dataset
        st.write("**Demo với Titanic Dataset:**")
        
        try:
            df = pd.read_csv('datasets/titanic.csv')
            st.write("Dataset preview:")
            st.dataframe(df.head(), use_container_width=True)
            
            n_samples = st.slider("Số lượng samples:", 5, 50, 10)
            
            if st.button("🚀 Extract Features from Titanic"):
                extractor = DictFeatureExtractor()
                dict_data, features, feature_names = extractor.demo_with_titanic(df, n_samples)
                
                st.success(f"✅ Đã extract {n_samples} samples!")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.write("**Original Data (dict format):**")
                    st.json(dict_data[:3])  # Show first 3
                
                with col_b:
                    st.write("**Feature Names:**")
                    st.write(list(feature_names))
                
                st.write("**Feature Matrix:**")
                df_features = pd.DataFrame(features, columns=feature_names)
                st.dataframe(df_features, use_container_width=True)
                
                st.write(f"📏 **Shape:** {features.shape}")
        
        except FileNotFoundError:
            st.error("❌ Không tìm thấy datasets/titanic.csv. Vui lòng chạy download_data.py trước!")

# ==================== FEATURE HASHING ====================
elif page == "# Feature Hashing":
    st.header("7.2.2. Feature Hashing")
    
    st.subheader("📚 Lý thuyết")
    st.write("""
    **Feature Hashing** (Hashing Trick) sử dụng hash function để chuyển features thành vector 
    có kích thước cố định.
    
    **Ưu điểm:**
    - ⚡ Rất nhanh (không cần lưu vocabulary)
    - 💾 Tiết kiệm bộ nhớ
    - 🔄 Xử lý được unseen features
    
    **Nhược điểm:**
    - ⚠️ Hash collision (nhiều features → cùng 1 hash value)
    - ❓ Không biết feature gốc là gì (one-way)
    """)
    
    st.markdown("---")
    st.subheader("🎮 Demo Interactive")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        country1 = st.text_input("Country 1:", "Vietnam")
        city1 = st.text_input("City 1:", "Hanoi")
        age1 = st.number_input("Age 1:", 18, 100, 25, key="hash_age1")
    
    with col2:
        country2 = st.text_input("Country 2:", "Thailand")
        city2 = st.text_input("City 2:", "Bangkok")
        age2 = st.number_input("Age 2:", 18, 100, 28, key="hash_age2")
    
    with col3:
        country3 = st.text_input("Country 3:", "Vietnam")
        city3 = st.text_input("City 3:", "HCM")
        age3 = st.number_input("Age 3:", 18, 100, 30, key="hash_age3")
    
    n_features = st.slider("Số features sau khi hash:", 5, 20, 10)
    
    if st.button("🚀 Extract & Compare"):
        data = [
            {'country': country1, 'city': city1, 'age': age1},
            {'country': country2, 'city': city2, 'age': age2},
            {'country': country3, 'city': city3, 'age': age3}
        ]
        
        extractor = HashFeatureExtractor(n_features=n_features)
        comparison, dict_features, hash_features = extractor.compare_with_dict_vectorizer(data)
        
        st.success("✅ Extraction hoàn tất!")
        
        # Show original data
        st.write("**Input Data:**")
        st.json(data)
        
        # Comparison table
        st.subheader("📊 So sánh DictVectorizer vs FeatureHasher")
        
        comp_df = pd.DataFrame({
            'Method': ['DictVectorizer', 'FeatureHasher'],
            'Number of Features': [comparison['DictVectorizer']['n_features'], 
                                   comparison['FeatureHasher']['n_features']],
            'Memory (bytes)': [comparison['DictVectorizer']['memory_size'],
                              comparison['FeatureHasher']['memory_size']]
        })
        st.dataframe(comp_df, use_container_width=True)
        
        # Show features
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.write("**DictVectorizer Output:**")
            st.write(dict_features)
            st.caption(f"Shape: {dict_features.shape}")
        
        with col_b:
            st.write("**FeatureHasher Output:**")
            st.write(hash_features)
            st.caption(f"Shape: {hash_features.shape}")
        
        st.info("""
        **💡 Nhận xét:**
        - FeatureHasher có số features cố định (bạn chọn)
        - DictVectorizer có số features = số unique values
        - FeatureHasher tiết kiệm bộ nhớ hơn khi có nhiều categorical values
        """)

# ==================== TEXT FEATURES ====================
elif page == "📝 Text Features":
    st.header("7.2.3. Text Feature Extraction")
    
    st.subheader("📚 Lý thuyết")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        **Count Vectorizer:**
        - Đếm số lần xuất hiện của mỗi từ
        - Ví dụ: "I love ML" → [1, 1, 1]
        
        **Ưu điểm:**
        - Đơn giản, dễ hiểu
        - Phù hợp với short texts
        """)
    
    with col2:
        st.write("""
        **TF-IDF Vectorizer:**
        - TF (Term Frequency): Tần suất từ trong document
        - IDF (Inverse Document Frequency): Độ quan trọng của từ
        - TF-IDF = TF × IDF
        
        **Ưu điểm:**
        - Phản ánh tầm quan trọng của từ
        - Giảm trọng số của từ phổ biến (the, is, a,...)
        """)
    
    st.markdown("---")
    st.subheader("🎮 Demo Interactive")
    
    # Mode selection
    demo_mode = st.radio("Chọn mode:", ["Nhập text tự do", "Mẫu có sẵn"], horizontal=True)
    
    if demo_mode == "Nhập text tự do":
        st.write("**Nhập các đoạn text (mỗi dòng = 1 document):**")
        
        text_input = st.text_area(
            "Your texts:",
            value="I love machine learning\nPython is great for data science\nDeep learning requires GPUs",
            height=150
        )
        
        texts = [t.strip() for t in text_input.split('\n') if t.strip()]
    
    else:
        texts = [
            "Machine learning is a subset of artificial intelligence",
            "Data science combines statistics and programming",
            "Python is the most popular language for machine learning",
            "Deep learning uses neural networks with multiple layers",
            "Natural language processing deals with text and speech"
        ]
        st.write("**Sample texts:**")
        for i, text in enumerate(texts, 1):
            st.write(f"{i}. {text}")
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        method = st.selectbox("Method:", ["tfidf", "count"])
    with col2:
        max_features = st.slider("Max features:", 10, 50, 20)
    with col3:
        show_top = st.slider("Show top words:", 5, 20, 10)
    
    if st.button("🚀 Extract Text Features"):
        extractor = TextFeatureExtractor(method=method, max_features=max_features)
        features, feature_names, top_words = extractor.analyze_text(texts)
        
        st.success("✅ Extraction hoàn tất!")
        
        # Feature matrix
        st.subheader("📊 Feature Matrix")
        df_features = pd.DataFrame(features, columns=feature_names)
        df_features.index = [f"Doc {i+1}" for i in range(len(texts))]
        st.dataframe(df_features, use_container_width=True)
        
        st.write(f"📏 **Shape:** {features.shape} ({len(texts)} documents × {features.shape[1]} words)")
        
        # Top words
        col_a, col_b = st.columns([1, 2])
        
        with col_a:
            st.subheader(f"🔝 Top {show_top} Words")
            top_df = pd.DataFrame(top_words[:show_top], columns=['Word', 'Score'])
            st.dataframe(top_df, use_container_width=True)
        
        with col_b:
            st.subheader("📈 Visualization")
            fig = extractor.visualize_features(texts, feature_names, features)
            st.plotly_chart(fig, use_container_width=True)
        
        # Explanation
        with st.expander("💡 Giải thích kết quả"):
            if method == 'tfidf':
                st.write("""
                **TF-IDF scores cao** = từ quan trọng trong document và ít xuất hiện trong các documents khác
                
                Ví dụ:
                - "learning", "machine" có score cao vì xuất hiện nhiều lần
                - "the", "is", "a" bị loại bỏ (stopwords) hoặc có score thấp
                """)
            else:
                st.write("""
                **Count values** = số lần từ xuất hiện trong mỗi document
                
                - Count = 3: từ xuất hiện 3 lần
                - Count = 0: từ không xuất hiện
                """)

# ==================== IMAGE FEATURES ====================
elif page == "🖼️ Image Features":
    st.header("7.2.4. Image Feature Extraction")
    
    st.subheader("📚 Lý thuyết")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("""
        **Color Histogram:**
        - Đếm số lượng pixels cho mỗi màu
        - 3 histograms: Red, Green, Blue
        
        **Ứng dụng:**
        - Image similarity
        - Object tracking
        """)
    
    with col2:
        st.write("""
        **HOG (Histogram of Oriented Gradients):**
        - Phát hiện edges và hướng của chúng
        - Bất biến với lighting
        
        **Ứng dụng:**
        - Object detection
        - Face recognition
        """)
    
    with col3:
        st.write("""
        **Edge Detection:**
        - Tìm biên của objects
        - Sử dụng Canny algorithm
        
        **Ứng dụng:**
        - Shape detection
        - Image segmentation
        """)
    
    st.markdown("---")
    st.subheader("🎮 Demo Interactive")
    
    # Upload image
    uploaded_file = st.file_uploader("Upload ảnh của bạn:", type=['png', 'jpg', 'jpeg'])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        method = st.radio("Chọn phương pháp:", ["histogram", "hog", "edges"])
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        
        with col2:
            st.write("**Original Image:**")
            st.image(image, width=300)
        
        if st.button("🚀 Extract Image Features"):
            extractor = ImageFeatureExtractor(method=method)
            
            with st.spinner("Đang xử lý..."):
                features = extractor.extract_features(image)
                
                st.success("✅ Extraction hoàn tất!")
                
                # Show features
                st.subheader("📊 Extracted Features")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.write(f"**Feature Vector Shape:** {features.shape}")
                    st.write(f"**Number of Features:** {len(features)}")
                    
                    st.write("**First 20 features:**")
                    st.write(features[:20])
                
                with col_b:
                    # Visualization
                    if method in ['histogram', 'edges']:
                        fig = extractor.visualize_features(image)
                        st.pyplot(fig)
                    elif method == 'hog':
                        st.write("**Feature Distribution:**")
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(10, 3))
                        ax.plot(features[:100])
                        ax.set_title('First 100 HOG Features')
                        ax.set_xlabel('Feature Index')
                        ax.set_ylabel('Value')
                        st.pyplot(fig)
                
                # Download features
                st.download_button(
                    label="📥 Download Feature Vector",
                    data=features.tobytes(),
                    file_name=f"{method}_features.npy",
                    mime="application/octet-stream"
                )
                
                # Explanation
                with st.expander("💡 Giải thích kết quả"):
                    if method == 'histogram':
                        st.write("""
                        **Color Histogram** cho thấy phân bố màu sắc trong ảnh:
                        - Peaks cao = nhiều pixels có màu đó
                        - 3 histograms riêng biệt cho R, G, B channels
                        - Normalized về [0, 1] để dễ so sánh
                        """)
                    elif method == 'hog':
                        st.write("""
                        **HOG Features** mô tả shape và structure của objects:
                        - Tính gradient direction tại mỗi pixel
                        - Chia ảnh thành cells và tính histogram
                        - Feature vector dài (thường >1000 dimensions)
                        """)
                    elif method == 'edges':
                        st.write("""
                        **Edge Features** highlight biên của objects:
                        - Sử dụng Canny edge detector
                        - Giá trị 1 = edge, 0 = không phải edge
                        - Flattened thành vector 1D
                        """)
    
    else:
        st.info("👆 Upload một ảnh để bắt đầu!")
        
        # Show example
        st.write("**Hoặc thử với ảnh mẫu:**")
        if st.button("Sử dụng ảnh mẫu"):
            # Create sample image
            sample_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            st.image(sample_image, caption="Sample Image", width=300)
            st.info("Bạn có thể upload ảnh riêng của bạn ở trên!")

# ==================== SO SÁNH ====================
elif page == "🔬 So sánh":
    st.header("So sánh 4 Phương pháp Feature Extraction")
    
    st.subheader("📊 Bảng so sánh tổng quan")
    
    comparison_data = {
        'Phương pháp': ['Dict Features', 'Feature Hashing', 'Text Features', 'Image Features'],
        'Input Type': ['Dictionary', 'Dictionary', 'Text', 'Image'],
        'Output Type': ['Dense Vector', 'Dense Vector', 'Sparse Vector', 'Dense Vector'],
        'Số Features': ['Tự động (= unique values)', 'Cố định (user định)', 'User định (max_features)', 'Phụ thuộc method'],
        'Tốc độ': ['Nhanh', 'Rất nhanh ⚡', 'Trung bình', 'Chậm'],
        'Bộ nhớ': ['Trung bình', 'Thấp 💾', 'Cao', 'Rất cao'],
        'Ứng dụng': ['Categorical data', 'Large vocabulary', 'NLP, Text Mining', 'Computer Vision']
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)
    
    st.markdown("---")
    
    # Ưu nhược điểm
    st.subheader("⚖️ Ưu & Nhược điểm")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**1️⃣ Dict Features (DictVectorizer)**")
        st.success("""
        ✅ **Ưu điểm:**
        - Dễ hiểu, interpretable
        - Giữ nguyên feature names
        - Không bị hash collision
        """)
        st.error("""
        ❌ **Nhược điểm:**
        - Tốn bộ nhớ với large vocabulary
        - Cần fit trước khi transform
        - Không xử lý được unseen values
        """)
        
        st.write("**2️⃣ Feature Hashing**")
        st.success("""
        ✅ **Ưu điểm:**
        - Rất nhanh, scalable
        - Không cần lưu vocabulary
        - Xử lý được unseen values
        - Tiết kiệm bộ nhớ
        """)
        st.error("""
        ❌ **Nhược điểm:**
        - Hash collision
        - Mất interpretability
        - Không thể reverse
        """)
    
    with col2:
        st.write("**3️⃣ Text Features (TF-IDF)**")
        st.success("""
        ✅ **Ưu điểm:**
        - Phản ánh importance của từ
        - Giảm noise (stopwords)
        - Hiệu quả cho text classification
        """)
        st.error("""
        ❌ **Nhược điểm:**
        - Mất thứ tự từ (bag of words)
        - Không hiểu ngữ nghĩa
        - Sparse vector (tốn memory)
        """)
        
        st.write("**4️⃣ Image Features**")
        st.success("""
        ✅ **Ưu điểm:**
        - Capture visual information
        - Robust với transformations
        - Nhiều methods lựa chọn
        """)
        st.error("""
        ❌ **Nhược điểm:**
        - Tính toán chậm
        - High dimensional
        - Cần preprocessing
        """)
    
    st.markdown("---")
    
    # Performance comparison
    st.subheader("⚡ So sánh Performance")
    
    perf_data = {
        'Method': ['Dict', 'Hash', 'Text (TF-IDF)', 'Image (HOG)'],
        'Training Time': [0.01, 0.005, 0.5, 2.0],
        'Inference Time': [0.005, 0.002, 0.1, 1.5],
        'Memory Usage (MB)': [10, 5, 50, 100]
    }
    
    df_perf = pd.DataFrame(perf_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Training Time (seconds):**")
        st.bar_chart(df_perf.set_index('Method')['Training Time'])
    
    with col2:
        st.write("**Memory Usage (MB):**")
        st.bar_chart(df_perf.set_index('Method')['Memory Usage (MB)'])
    
    st.caption("*Số liệu mang tính chất minh họa với dataset nhỏ")
    
    st.markdown("---")
    
    # Khi nào dùng cái gì
    st.subheader("🎯 Khi nào nên dùng phương pháp nào?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Dùng Dict Features khi:**
        - Dataset nhỏ, số features không quá lớn
        - Cần interpretability
        - Có categorical data
        - Ví dụ: Dữ liệu khách hàng, sản phẩm
        """)
        
        st.info("""
        **Dùng Feature Hashing khi:**
        - Dataset rất lớn (millions of features)
        - Cần speed & scalability
        - Không quan tâm interpretability
        - Ví dụ: Click-through rate prediction
        """)
    
    with col2:
        st.info("""
        **Dùng Text Features khi:**
        - Làm việc với văn bản
        - Text classification, sentiment analysis
        - Có đủ bộ nhớ
        - Ví dụ: Spam detection, topic modeling
        """)
        
        st.info("""
        **Dùng Image Features khi:**
        - Làm việc với ảnh
        - Computer vision tasks
        - Có GPU (cho deep learning)
        - Ví dụ: Object detection, face recognition
        """)
    
    st.markdown("---")
    
    # Summary
    st.subheader("📝 Tóm tắt")
    st.write("""
    **Không có phương pháp nào là tốt nhất cho mọi trường hợp!**
    
    Lựa chọn phương pháp phụ thuộc vào:
    1. **Loại dữ liệu:** Categorical, Text, Image?
    2. **Kích thước dataset:** Lớn hay nhỏ?
    3. **Yêu cầu:** Speed, Accuracy, Interpretability?
    4. **Resources:** Memory, CPU, GPU?
    
    💡 **Best practice:** Thử nhiều methods và so sánh kết quả!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🎓 Demo by Group 8 - Môn Khai Phá Dữ Liệu</p>
    <p>Made with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
