import torch
import torch.nn as nn
import torch.nn.functional as F

def bcecc_loss(logits, labels, embeddings, alpha=0.9, tau=0.4, beta_pos=1.0):
    """
    Args:
        embeddings: 样本的嵌入向量 (N, D) (N为batch大小, D为嵌入维度)
        labels: 样本标签 (N, )
        logits: 线性层输出的logits (N, )
    Returns:
        loss_cc: 综合损失 L_CC
    """
    # ========== 1. 计算加权二分类交叉熵损失 L_BCE ==========
    bce_loss = F.binary_cross_entropy_with_logits(
        logits.squeeze(), labels.float(), pos_weight=torch.tensor(beta_pos), reduction='sum'
    )
    
    # ========== 2. 计算对比损失 L_Contr ==========
    N = embeddings.size(0)  # 样本数量
    # 归一化嵌入向量
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # 计算余弦相似度
    sim_matrix = torch.matmul(embeddings, embeddings.T) / tau  # (N, N)
    
    # 创建样本相似度掩码
    labels = labels.view(-1, 1)  # (N, 1)
    mask = torch.eq(labels, labels.T).float()  # 相同标签为1，否则为0
    mask_neg = 1 - mask  # 负样本掩码

    # 计算对比损失
    loss_contrastive = 0.0
    for i in range(N):
        positive_mask = mask[i]  # 与当前样本相同标签的样本
        negative_mask = mask_neg[i]  # 与当前样本不同标签的样本
        
        # 排除自己
        positive_mask[i] = 0
        negative_mask[i] = 0
        
        # 正样本和负样本相似度
        positives = torch.exp(sim_matrix[i][positive_mask > 0])
        negatives = torch.exp(sim_matrix[i][negative_mask > 0])
        
        # 对比损失的分子与分母
        pos_sum = torch.sum(positives)  # 正样本相似度之和
        neg_sum = torch.sum(negatives)  # 负样本相似度之和
        
        # 计算对比损失
        if pos_sum > 0:
            loss_contrastive += -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8))
    
    # loss_contrastive /= N  # 归一化对比损失
    
    # ========== 3. 结合两个损失函数 L_CC ==========
    loss_cc = alpha * bce_loss + (1 - alpha) * loss_contrastive

    # print("BCE Loss:", bce_loss.item(), "Contrastive Loss:", loss_contrastive.item(), "Total Loss (L_CC):", loss_cc.item())
    return loss_cc

# ========== 简单二分类网络定义 ==========
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(SimpleClassifier, self).__init__()
        self.embedding_layer = nn.Linear(input_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 1)
    
    def forward(self, x):
        embeddings = self.embedding_layer(x)
        logits = self.classifier(embeddings)
        return embeddings, logits

# ========== 测试代码 ==========
if __name__ == "__main__":
    # 模拟数据
    torch.manual_seed(42)
    batch_size = 8
    input_dim = 10
    embedding_dim = 4

    # 生成输入数据和标签
    inputs = torch.randn(batch_size, input_dim)
    labels = torch.randint(0, 2, (batch_size,))  # 二分类标签 0 或 1

    # 初始化网络和损失函数
    model = SimpleClassifier(input_dim, embedding_dim)
    criterion = CustomLoss(alpha=0.9, tau=0.4, beta_pos=1.0)

    # 前向传播
    embeddings, logits = model(inputs)

    # 计算损失
    loss = criterion(embeddings, labels, logits)
    print("Total Loss (L_CC):", loss.item())
