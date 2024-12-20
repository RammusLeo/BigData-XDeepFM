import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder  # 导入 LabelEncoder
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
sns.countplot(x='label', data=df, palette="Set2")  # 使用更鲜艳的配色
plt.title('Label Distribution', fontsize=16)
plt.xlabel('Label', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.savefig('./visual/label_distribution.png', dpi=300)  # 保存为PNG图片到 ./visual 文件夹
plt.close()

# 2. 数值特征分布（以I1为例）
plt.figure(figsize=(6, 4))
sns.histplot(df['I1'].dropna(), bins=30, kde=True, color='teal')  # 设置颜色为青色
plt.title('Distribution of I1', fontsize=16)
plt.xlabel('I1', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.savefig('./visual/distribution_of_I1.png', dpi=300)  # 保存为PNG图片到 ./visual 文件夹
plt.close()

# 3. 特征间相关性分析
# 只选择数值型特征列进行相关性计算
numeric_cols = df.select_dtypes(include=['number']).columns  # 选择数值型列
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, linecolor='black')
plt.title('Correlation Matrix of Numerical Features', fontsize=16)
plt.savefig('./visual/correlation_matrix.png', dpi=300)  # 保存为PNG图片到 ./visual 文件夹
plt.close()

# # 4. 类别特征频率分析（以C1为例）
# plt.figure(figsize=(8, 6))

# # Use a higher contrast color palette and increase the bar width
# sns.countplot(y='C1', data=df, palette="Set1", width=0.8)  # Set1 for better contrast and adjust bar width

# plt.title('Frequency of Categories in C1', fontsize=18, fontweight='bold')  # Increase title size and make it bold
# plt.xlabel('Frequency', fontsize=14)
# plt.ylabel('Category (C1)', fontsize=14)

# # Rotate the y-axis labels for better readability (if category labels are long)
# plt.yticks(rotation=0, fontsize=12)

# # Add grid lines for better clarity
# plt.grid(axis='x', linestyle='--', alpha=0.6)

# # Remove the y-axis label and ticks (already done)
# plt.ylabel('')  # Remove the y-axis label
# plt.yticks([])  # Remove y-axis ticks

# # Save the figure with higher resolution
# plt.savefig('./visual/frequency_of_categories_in_C1.png', dpi=300, bbox_inches='tight')  # bbox_inches='tight' to avoid cutting labels
# plt.close()

# 4. 类别特征频率分析（以C1为例）
# 将类别数据编码为数值型数据
le = LabelEncoder()
df['C1_encoded'] = le.fit_transform(df['C1'])

plt.figure(figsize=(8, 6))

# 使用 kdeplot 绘制类别的分布拟合曲线
sns.kdeplot(data=df['C1_encoded'], fill=True, color='#B40426', linewidth=3, alpha=0.5)

# # 设置对数坐标轴
# plt.xscale('log')
# plt.yscale('log')

# 添加标题和标签
plt.title('Log-Scaled Frequency Distribution of Categories in C1', fontsize=16)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# 保存为PNG图片到 ./visual 文件夹
plt.savefig('./visual/frequency_of_categories_in_C1_log_scaled.png', dpi=300)  
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
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=df['label'], palette='plasma', s=100, edgecolor='black', alpha=0.7)  # 增加点的大小，调整透明度
plt.title('PCA of Numerical Features', fontsize=16)
plt.xlabel('PCA Component 1', fontsize=12)
plt.ylabel('PCA Component 2', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Label', title_fontsize='13', loc='upper right')
plt.savefig('./visual/PCA_of_Numerical_Features.png', dpi=300)  # 保存为PNG图片到 ./visual 文件夹
plt.close()

# python ./visual/data_arch.py