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

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    h1 {
        color: #FF4B4B;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

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

with st.expander("👥 Thông tin nhóm", expanded=False):
    st.write("**Nhóm 8 - Môn Khai Phá Dữ Liệu**")
    
    team_members = [
        {'STT': 1, 'Họ tên': 'Lò Văn Bằng', 'MSSV': '2251061721', 'Phần đảm nhận': 'Image Features'},
        {'STT': 2, 'Họ tên': 'Nguyễn Trung Kiên', 'MSSV': '2251061811', 'Phần đảm nhận': 'Dict Features'},
        {'STT': 3, 'Họ tên': 'Thiều Bá Việt', 'MSSV': '2251061924', 'Phần đảm nhận': 'Text Features'},
        {'STT': 4, 'Họ tên': 'Lường Văn Cương', 'MSSV': '20210004', 'Phần đảm nhận': 'Feature Hashing'}
    ]
    
    df_team = pd.DataFrame(team_members)
    st.dataframe(df_team, hide_index=True, use_container_width=True)

# Title
st.title("🚀 Feature Extraction Demo")
st.markdown("**Nhóm 8 - Môn Khai Phá Dữ Liệu**")
st.markdown("---")

# Sidebar
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio(
    "Chọn phương pháp:",
    ["🏠 Tổng quan", "📊 Dict Features", "# Feature Hashing", 
     "📝 Text Features", "🖼️ Image Features", "🔬 So sánh", "🎯 ML Performance"]
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
    st.markdown("""
    **DictVectorizer** chuyển đổi dữ liệu dạng dictionary thành feature vectors.
    
    **Ví dụ:** """)
    st.code("""
    Input:  [{'age': '25', 'city': 'Hanoi'}, {'age': '30', 'city': 'Danang'}]
    Output: [[25, 1, 0], [30, 0, 1]]  # [age, city=Hanoi, city=Danang]""",language='python')

    st.markdown("""
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
        method = st.selectbox("Method:", ["tfidf", "count", "hashing"])
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

# ==================== IMAGE FEATURES (SCIKIT-LEARN VERSION) ====================
elif page == "🖼️ Image Features":
    st.header("7.2.4. Image Feature Extraction (Scikit-learn)")
    
    st.info("⚠️ **Theo tài liệu scikit-learn**, Image Feature Extraction bao gồm: **Patch Extraction** và **Image-to-Graph Conversion**")
    
    st.subheader("📚 Lý thuyết")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        **1️⃣ Patch Extraction:**
        - Chia ảnh thành patches nhỏ
        - Học local features từ raw pixels
        - API: `extract_patches_2d()`, `PatchExtractor`
        
        **Ứng dụng:**
        - Dictionary Learning
        - Image Denoising
        - Texture Analysis
        """)
    
    with col2:
        st.write("""
        **2️⃣ Image-to-Graph:**
        - Chuyển ảnh thành graph structure
        - Mỗi pixel = 1 node trong graph
        - API: `img_to_graph()`
        
        **Ứng dụng:**
        - Spectral Clustering
        - Image Segmentation
        - Region Analysis
        """)
    
    st.markdown("---")
    st.subheader("🎮 Demo Interactive")
    
    # Import sklearn version
    from utils.image_features import ImageFeatureExtractor
    
    # Upload or select image
    uploaded_file = st.file_uploader("Upload ảnh của bạn:", type=['png', 'jpg', 'jpeg'])
    
    st.write("**Hoặc chọn ảnh mẫu:**")
    sample_choice = st.selectbox(
        "Chọn ảnh mẫu:",
        ["Không chọn", "Red Image", "Green Image", "Pattern Image", "Gradient Image"]
    )
    
    # Xử lý ảnh
    image = None
    if sample_choice != "Không chọn":
        sample_map = {
            "Red Image": "datasets/sample_images/red_image.png",
            "Green Image": "datasets/sample_images/green_image.png",
            "Pattern Image": "datasets/sample_images/pattern_image.png",
            "Gradient Image": "datasets/sample_images/gradient_image.png"
        }
        
        sample_path = sample_map[sample_choice]
        if os.path.exists(sample_path):
            image = Image.open(sample_path)
        else:
            st.error(f"❌ Không tìm thấy ảnh mẫu: {sample_path}")
    
    elif uploaded_file is not None:
        image = Image.open(uploaded_file)
    
    # Method selection & parameters
    col1, col2 = st.columns([1, 2])
    
    with col1:
        method = st.radio("Chọn phương pháp:", ["patches", "graph"])
        
        if method == "patches":
            st.write("**Tham số Patch Extraction:**")
            patch_size = st.slider("Patch size:", 16, 64, 32, 8)
            max_patches = st.slider("Max patches:", 20, 200, 100, 10)
        else:
            st.write("**Tham số Graph Clustering:**")
            n_clusters = st.slider("Number of clusters:", 2, 8, 3)
    
    if image is not None:
        with col2:
            st.write("**Original Image:**")
            st.image(image, width=300)
            image_array = np.array(image)
            st.caption(f"Size: {image_array.shape[1]}×{image_array.shape[0]} pixels")
        
        if st.button("🚀 Extract Features (scikit-learn)"):
            if method == "patches":
                # ========== PATCH EXTRACTION ==========
                extractor = ImageFeatureExtractor(method='patches')
                
                with st.spinner("Extracting patches..."):
                    features, extras = extractor.extract_features(
                        image,
                        patch_size=(patch_size, patch_size),
                        max_patches=max_patches
                    )
                    
                    st.success("✅ Patch Extraction hoàn tất!")
                    
                    # Show info
                    st.subheader("📊 Kết Quả Patch Extraction")
                    
                    # Metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Number of Patches", extras['n_patches'])
                    with col_b:
                        st.metric("Patch Size", f"{patch_size}×{patch_size}")
                    with col_c:
                        st.metric("Features per Patch", patch_size * patch_size * 3)
                    
                    # Details
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.write("**📋 Thông tin chi tiết:**")
                        st.write(f"- Patches shape: `{extras['patches'].shape}`")
                        st.write(f"- Flattened features: `{features.shape}`")
                        st.write(f"- Total features: `{features.shape[0] * features.shape[1]:,}`")
                        
                        st.write("\n**🔢 First patch (first 20 features):**")
                        st.write(features[0][:20])
                        
                        # Explain patch calculation
                        with st.expander("💡 Cách tính số patches"):
                            st.write(f"""
                            **Phương pháp: Random Sampling**
                            
                            - Ảnh gốc: {image_array.shape[1]}×{image_array.shape[0]}
                            - Patch size: {patch_size}×{patch_size}
                            - `max_patches={max_patches}` → Random sampling {max_patches} vị trí
                            
                            **Nếu cắt đều (non-overlapping):**
                            - Số patches ngang: {image_array.shape[1] // patch_size}
                            - Số patches dọc: {image_array.shape[0] // patch_size}
                            - Tổng: {(image_array.shape[1] // patch_size) * (image_array.shape[0] // patch_size)} patches
                            
                            **Với random sampling:**
                            - Chọn ngẫu nhiên {max_patches} vị trí
                            - Có thể overlap (chồng lấn)
                            - Linh hoạt hơn cho Dictionary Learning
                            """)
                    
                    with col_b:
                        st.write("**🖼️ Visualization (16 patches đầu tiên):**")
                        fig = extractor.visualize_patches(extras['patches'], n_display=16)
                        st.pyplot(fig)
                    
                    # Explanation
                    with st.expander("📖 Giải thích Patch Extraction"):
                        st.write("""
                        **Patch Extraction là gì?**
                        
                        Chia ảnh lớn thành nhiều patches (mảnh) nhỏ để học các local patterns.
                        
                        **Quy trình:**
                        1. Chọn patch size (ví dụ: 32×32)
                        2. Random sampling hoặc sliding window
                        3. Extract từng patch thành vector
                        4. Flatten: 32×32×3 = 3,072 features/patch
                        
                        **Ứng dụng thực tế:**
                        - **Dictionary Learning**: Học "alphabet" của hình ảnh
                        - **Image Denoising**: Khử nhiễu bằng cách so sánh patches
                        - **Texture Recognition**: Phân loại textures (gỗ, vải, đá...)
                        - **Feature Extraction**: Dùng làm input cho ML models
                        
                        **Tham khảo:**
                        - [sklearn.feature_extraction.image.extract_patches_2d](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.extract_patches_2d.html)
                        """)
            
            else:
                # ========== IMAGE-TO-GRAPH ==========
                extractor = ImageFeatureExtractor(method='graph')
                
                with st.spinner("Converting to graph & clustering..."):
                    labels, extras = extractor.extract_features(
                        image,
                        n_clusters=n_clusters
                    )
                    
                    st.success("✅ Image-to-Graph Conversion hoàn tất!")
                    
                    # Show info
                    st.subheader("📊 Kết Quả Graph-based Segmentation")
                    
                    # Metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Number of Nodes", extras['graph'].shape[0])
                    with col_b:
                        st.metric("Number of Clusters", n_clusters)
                    with col_c:
                        st.metric("Graph Edges", extras['graph'].nnz // 2)
                    
                    # Details
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.write("**📋 Thông tin Graph:**")
                        st.write(f"- Graph shape: `{extras['graph'].shape}`")
                        st.write(f"- Graph type: Sparse adjacency matrix")
                        st.write(f"- Labels shape: `{labels.shape}`")
                        st.write(f"- Segmented shape: `{extras['segmented'].shape}`")
                        
                        st.write("\n**📊 Cluster Distribution:**")
                        unique, counts = np.unique(labels, return_counts=True)
                        cluster_data = []
                        for cluster, count in zip(unique, counts):
                            percentage = (count / len(labels)) * 100
                            cluster_data.append({
                                'Cluster': cluster,
                                'Pixels': count,
                                'Percentage': f"{percentage:.1f}%"
                            })
                        st.dataframe(pd.DataFrame(cluster_data), hide_index=True, use_container_width=True)
                        
                        with st.expander("💡 Tại sao resize về 50×50?"):
                            st.write("""
                            Graph-based clustering **rất chậm** với ảnh lớn vì:
                            - Ảnh 256×256 = 65,536 nodes → Ma trận 65,536 × 65,536!
                            - Spectral clustering complexity: O(n³)
                            
                            Resize về 50×50:
                            - 2,500 nodes → Nhanh hơn nhiều
                            - Vẫn giữ được structure chính của ảnh
                            - Phù hợp cho demo & education
                            """)
                    
                    with col_b:
                        st.write("**🖼️ Segmentation Result:**")
                        fig = extractor.visualize_segmentation(
                            image_array,
                            extras['segmented'],
                            extras['small_image']
                        )
                        st.pyplot(fig)
                    
                    # Explanation
                    with st.expander("📖 Giải thích Image-to-Graph"):
                        st.write("""
                        **Image-to-Graph là gì?**
                        
                        Chuyển đổi hình ảnh thành cấu trúc đồ thị để phân tích relationships giữa các pixels.
                        
                        **Cấu trúc Graph:**
                        - **Nodes**: Mỗi pixel = 1 node
                        - **Edges**: Kết nối với 4 hoặc 8 neighbors
                        - **Weights**: Dựa trên độ khác biệt màu sắc
                        
                        **Spectral Clustering:**
                        1. Build graph từ ảnh
                        2. Tính eigenvectors của Laplacian matrix
                        3. Clustering trong eigenspace
                        4. Gán labels về pixels
                        
                        **Ứng dụng thực tế:**
                        - **Medical Imaging**: Phân vùng cơ quan, tumor
                        - **Image Segmentation**: Tách object khỏi background
                        - **Region Analysis**: Phân tích từng vùng riêng biệt
                        - **Interactive Selection**: Click để select region
                        
                        **Ưu điểm:**
                        - Unsupervised (không cần labels)
                        - Tự động tìm boundaries
                        - Consider cả color và spatial proximity
                        
                        **Tham khảo:**
                        - [sklearn.feature_extraction.image.img_to_graph](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.img_to_graph.html)
                        - [Spectral Clustering](https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering)
                        """)
    
    else:
        st.info("👆 Upload một ảnh hoặc chọn ảnh mẫu để bắt đầu!")
        
        # Documentation links
        st.markdown("---")
        st.write("**📚 Tài liệu tham khảo chính thức:**")
        st.markdown("""
        - [Scikit-learn Feature Extraction Documentation](https://scikit-learn.org/stable/modules/feature_extraction.html#image-feature-extraction)
        - [`sklearn.feature_extraction.image.extract_patches_2d`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.extract_patches_2d.html)
        - [`sklearn.feature_extraction.image.PatchExtractor`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.PatchExtractor.html)
        - [`sklearn.feature_extraction.image.img_to_graph`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.img_to_graph.html)
        - [Spectral Clustering](https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering)
        """)
        
        st.write("**💡 Lưu ý:**")
        st.info("""
        Đây là Image Feature Extraction theo **tài liệu scikit-learn chính thức**, 
        khác với Computer Vision truyền thống (Color Histogram, HOG, SIFT, CNN features).
        """)

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

# ==================== ML PERFORMANCE COMPARISON ====================
elif page == "🎯 ML Performance":
    st.header("🎯 So sánh Performance với Machine Learning")
    
    st.write("""
    Trang này demo **hiệu suất thực tế** của các phương pháp feature extraction 
    khi kết hợp với Machine Learning models.
    """)
    
    st.markdown("---")
    
    # Demo 1: Dict vs Hash với Titanic
    st.subheader("1️⃣ Dict Features vs Feature Hashing (Titanic Dataset)")
    
    if st.button("🚀 Chạy so sánh Dict vs Hash"):
        with st.spinner("Đang training models..."):
            try:
                df = pd.read_csv('datasets/titanic.csv')
                
                # Chuẩn bị data
                df_clean = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']].dropna()
                df_clean['Age'] = df_clean['Age'].astype(int)
                df_clean['Fare'] = df_clean['Fare'].astype(int)
                
                # Convert to dict
                dict_data = df_clean[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].to_dict('records')
                labels = df_clean['Survived'].values
                
                # Import comparison
                from utils.model_comparison import FeatureExtractionComparison
                comparator = FeatureExtractionComparison()
                
                results = comparator.compare_dict_vs_hash(dict_data, labels)
                
                st.success("✅ Hoàn tất!")
                
                # Hiển thị kết quả
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("DictVectorizer Accuracy", f"{results['DictVectorizer']['accuracy']:.2%}")
                    st.write(f"⏱️ Transform time: {results['DictVectorizer']['transform_time']:.4f}s")
                    st.write(f"🏋️ Train time: {results['DictVectorizer']['train_time']:.4f}s")
                    st.write(f"📊 Features: {results['DictVectorizer']['n_features']}")
                    st.write(f"💾 Memory: {results['DictVectorizer']['memory_mb']:.2f} MB")
                
                with col2:
                    st.metric("FeatureHasher Accuracy", f"{results['FeatureHasher']['accuracy']:.2%}")
                    st.write(f"⏱️ Transform time: {results['FeatureHasher']['transform_time']:.4f}s")
                    st.write(f"🏋️ Train time: {results['FeatureHasher']['train_time']:.4f}s")
                    st.write(f"📊 Features: {results['FeatureHasher']['n_features']}")
                    st.write(f"💾 Memory: {results['FeatureHasher']['memory_mb']:.2f} MB")
                
                # Chart
                import plotly.graph_objects as go
                
                fig = go.Figure(data=[
                    go.Bar(name='DictVectorizer', x=['Accuracy', 'Speed (1/time)', 'Memory Efficiency'], 
                           y=[results['DictVectorizer']['accuracy'], 
                              1/results['DictVectorizer']['transform_time'],
                              1/results['DictVectorizer']['memory_mb']]),
                    go.Bar(name='FeatureHasher', x=['Accuracy', 'Speed (1/time)', 'Memory Efficiency'], 
                           y=[results['FeatureHasher']['accuracy'],
                              1/results['FeatureHasher']['transform_time'],
                              1/results['FeatureHasher']['memory_mb']])
                ])
                
                fig.update_layout(barmode='group', title='Performance Comparison')
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **💡 Nhận xét:**
                - DictVectorizer thường có accuracy cao hơn (nhiều features hơn)
                - FeatureHasher nhanh hơn và tiết kiệm bộ nhớ hơn
                - Trade-off: Accuracy vs Speed/Memory
                """)
                
            except FileNotFoundError:
                st.error("❌ Không tìm thấy datasets/titanic.csv")
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
    
    st.markdown("---")
    
    # Demo 2: Count vs TF-IDF
    st.subheader("2️⃣ Count vs TF-IDF (Text Classification)")
    
    sample_texts = [
        "I love this product, it's amazing!",
        "Terrible quality, waste of money",
        "Best purchase ever, highly recommend",
        "Poor service, very disappointed",
        "Excellent value for money",
        "Not worth it, very bad",
        "Great experience, will buy again",
        "Horrible, do not buy",
        "Perfect, exactly what I needed",
        "Worst product ever"
    ]
    
    sample_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative
    
    st.write("**Sample data (Sentiment Analysis):**")
    for i, (text, label) in enumerate(zip(sample_texts[:5], sample_labels[:5])):
        st.write(f"{i+1}. [{'+' if label else '-'}] {text}")
    
    if st.button("🚀 Chạy so sánh Text Methods"):
        with st.spinner("Đang training..."):
            # Tạo thêm data để có đủ cho train/test split
            texts = sample_texts * 20  # 200 samples
            labels_expanded = sample_labels * 20
            
            from utils.model_comparison import FeatureExtractionComparison
            comparator = FeatureExtractionComparison()
            
            results = comparator.compare_text_methods(texts, labels_expanded)
            
            st.success("✅ Hoàn tất!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("CountVectorizer Accuracy", f"{results['CountVectorizer']['accuracy']:.2%}")
                st.write(f"⏱️ Time: {results['CountVectorizer']['transform_time']:.4f}s")
                st.write(f"📊 Features: {results['CountVectorizer']['n_features']}")
            
            with col2:
                st.metric("TfidfVectorizer Accuracy", f"{results['TfidfVectorizer']['accuracy']:.2%}")
                st.write(f"⏱️ Time: {results['TfidfVectorizer']['transform_time']:.4f}s")
                st.write(f"📊 Features: {results['TfidfVectorizer']['n_features']}")
            
            st.info("""
            **💡 Nhận xét:**
            - TF-IDF thường perform tốt hơn cho text classification
            - CountVectorizer đơn giản hơn, phù hợp với short texts
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🎓 Demo by Group 8 - Môn Khai Phá Dữ Liệu</p>
    <p>Made with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
