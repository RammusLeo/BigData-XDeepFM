import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 Seaborn 风格
sns.set(style="whitegrid")

# 读取数据
data = pd.read_csv('./data/train_sample_10p.txt', delimiter='\t', header=None)

# 假设标签在第一列（label）中
labels = data[0]

# 过滤出标签为 0 和 1 的样本
filtered_labels = labels[labels.isin([0, 1])]

# 统计标签0和1的数量
label_counts = filtered_labels.value_counts()

# 绘制柱状图
plt.figure(figsize=(8, 6))
ax = label_counts.plot(kind='bar', color=['#1f77b4', '#ff7f0e'], edgecolor='black', width=0.7)

# 设置标题和轴标签
plt.title('Distribution of Positive and Negative Samples', fontsize=16, fontweight='bold')
plt.xlabel('Label', fontsize=14)
plt.ylabel('Sample Count', fontsize=14)

# 设置刻度字体大小
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)

# 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 为每个柱子添加值标签
for p in ax.patches:
    ax.annotate(f'{p.get_height()}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                textcoords='offset points')

# 保存图表
plt.tight_layout()
plt.savefig('./visual/pos_neg_compare_enhanced.png')

# 输出正负样本的比例
total_samples = len(filtered_labels)
positive_samples = label_counts.get(1, 0)
negative_samples = label_counts.get(0, 0)
positive_ratio = positive_samples / total_samples
negative_ratio = negative_samples / total_samples

print(f"Positive samples: {positive_samples}, Negative samples: {negative_samples}")
print(f"Positive sample ratio: {positive_ratio:.2f}, Negative sample ratio: {negative_ratio:.2f}")
