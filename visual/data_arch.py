import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# 读取数据，假设数据是以空格分隔的
file_path = './data/train_sample_10000p.txt'

# 尝试用不同的分隔符加载数据（这里默认使用空格分隔，您可以根据实际情况修改分隔符）
df = pd.read_csv(file_path, sep=r'\s+', header=0)

# 创建 ./visual 文件夹，如果不存在的话
if not os.path.exists('./visual'):
    os.makedirs('./visual')

# 1. 标签分布
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df)
plt.title('Label Distribution')
plt.savefig('./visual/label_distribution.png', dpi=300)  # 保存为PNG图片到 ./visual 文件夹
plt.close()

# 2. 数值特征分布（以I1为例）
plt.figure(figsize=(6, 4))
sns.histplot(df['I1'].dropna(), bins=30, kde=True)
plt.title('Distribution of I1')
plt.savefig('./visual/distribution_of_I1.png', dpi=300)  # 保存为PNG图片到 ./visual 文件夹
plt.close()

# 3. 特征间相关性分析
# 只选择数值型特征列进行相关性计算
numeric_cols = df.select_dtypes(include=['number']).columns  # 选择数值型列
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.savefig('./visual/correlation_matrix.png', dpi=300)  # 保存为PNG图片到 ./visual 文件夹
plt.close()

# 4. 类别特征频率分析（以C1为例）
plt.figure(figsize=(6, 4))
sns.countplot(y='C1', data=df)
plt.title('Frequency of Categories in C1')
plt.savefig('./visual/frequency_of_categories_in_C1.png', dpi=300)  # 保存为PNG图片到 ./visual 文件夹
plt.close()

# 5. 使用PCA进行降维并可视化（假设我们只关注数值特征）
features = numeric_cols.tolist()  # 选择数值型特征列
df_features = df[features].dropna()  # 去除缺失值
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_features)

# 使用PCA降到2D
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)

# 可视化降维后的结果
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=df['label'], palette='coolwarm')
plt.title('PCA of Numerical Features')
plt.savefig('./visual/PCA_of_Numerical_Features.png', dpi=300)  # 保存为PNG图片到 ./visual 文件夹
plt.close()

# python ./visual/data_arch.py