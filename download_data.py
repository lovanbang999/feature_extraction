# download_data.py
import pandas as pd
import os

# Tạo thư mục datasets
os.makedirs('datasets', exist_ok=True)

# Download Titanic từ URL
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)
df.to_csv('datasets/titanic.csv', index=False)

print("✅ Đã tải Titanic dataset!")
print(f"Shape: {df.shape}")
print(df.head())
