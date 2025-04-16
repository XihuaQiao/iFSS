import torch
from torch import nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # 可学习的query矩阵，为每个头创建独立的query
        self.query = nn.Parameter(torch.randn(num_heads, self.head_dim))
        
    def forward(self, features):
        # features: [B, C, H, W]
        B, C, H, W = features.shape
        
        # 重塑特征以适应多头处理
        features_reshaped = features.view(B, self.num_heads, self.head_dim, H, W)
        
        # 为每个头计算注意力权重
        attention_weights = []
        for h in range(self.num_heads):
            # 扩展query以匹配特征维度
            query_h = self.query[h].view(1, self.head_dim, 1, 1).expand(B, self.head_dim, 1, 1)
            
            # 计算点积注意力
            attn = torch.sum(features_reshaped[:, h] * query_h, dim=1)  # [B, H, W]
            attn = F.softmax(attn.view(B, -1), dim=1).view(B, H, W)
            attention_weights.append(attn)
            
        # 合并所有头的注意力权重
        attention_weights = torch.stack(attention_weights, dim=1)  # [B, num_heads, H, W]
        
        return attention_weights


def compute_attention(query, key, scale_factor=None):
    """计算缩放点积注意力
    
    Args:
        query: 查询张量 [B, C, 1, 1]
        key: 键张量 [B, C, H, W]
        scale_factor: 缩放因子，默认为1/sqrt(C)
    
    Returns:
        attention_weights: 注意力权重 [B, H, W]
    """
    if scale_factor is None:
        scale_factor = 1.0 / math.sqrt(query.size(1))
    
    # 计算注意力分数
    query = query.view(query.size(0), query.size(1), 1, 1)
    attention_scores = torch.sum(key * query, dim=1) * scale_factor  # [B, H, W]
    
    # 应用softmax获取注意力权重
    attention_weights = F.softmax(attention_scores.view(attention_scores.size(0), -1), dim=1)
    attention_weights = attention_weights.view(attention_scores.size(0), 
                                              attention_scores.size(1),
                                              attention_scores.size(2))
    
    return attention_weights


class SpatialGating(nn.Module):
    def __init__(self, in_channels):
        super(SpatialGating, self).__init__()
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 计算空间门控值
        gate = self.spatial_gate(x)  # [B, 1, H, W]
        
        # 应用门控
        gated_x = x * gate
        
        return gated_x
    
class SpatialGatingDistllationLoss(nn.Module):
    def __init__(self, in_channels, temperature=1.0, device=None):
        super().__init__()

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        ).to(device)
        # self.spatial_weights = nn.parameter(torch.ones())
        self.crit = nn.CosineSimilarity(dim=1)
        self.temperature = temperature

    def forward(self, student, teacher):
        
        cosine_similarity = self.crit(student, teacher) / self.temperature
        gate = self.spatial_gate(torch.concat([student, teacher], dim=1))

        loss = (1 - cosine_similarity) * gate
        
        return loss.mean()


class MetaWeightingDistillationLoss(nn.Module):
    def __init__(self, feature_dim, device=None):
        super().__init__()
        # 使用极少量的元参数
        self.temperature = nn.Parameter(torch.ones(1) * 0.5).to(device)
        self.channel_importance = nn.Parameter(torch.ones(feature_dim) / feature_dim).to(device)
        # self.crit = nn.CosineSimilarity(dim=1)
        # self.temperature = temperature
        
    def forward(self, features, old_features):
        # 计算特征统计信息
        feat_mean = features.mean(dim=[0, 2, 3], keepdim=True)  # 1,C,1,1
        old_feat_mean = old_features.mean(dim=[0, 2, 3], keepdim=True)  # 1,C,1,1
        
        # 计算特征差异
        feat_diff = torch.abs(feat_mean - old_feat_mean).squeeze()  # C
        
        # 基于差异和重要性生成权重
        channel_weights = torch.softmax(self.channel_importance / (feat_diff + 1e-5) / self.temperature, dim=0)
        channel_weights = channel_weights.view(1, -1, 1, 1)

        similarity = F.cosine_similarity(features.flatten(2), old_features.flatten(2), dim=2)

        weighted_similarity = similarity * channel_weights.squeeze(-1).squeeze(-1)

        loss = 1.0 - weighted_similarity.mean()

        return loss



class DynamicGating(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(DynamicGating, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 生成门控网络参数的超网络
        self.param_generator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim * 2)  # 权重和偏置
        )
        
    def forward(self, x):
        # 生成门控网络的参数
        params = self.param_generator(x.mean(dim=[2, 3]))
        weight, bias = params.split(self.feature_dim, dim=1)
        
        # 重塑参数
        weight = weight.view(-1, self.feature_dim, 1, 1)
        bias = bias.view(-1, self.feature_dim, 1, 1)
        
        # 应用动态门控
        gate = torch.sigmoid(torch.sum(x * weight, dim=1, keepdim=True) + bias)
        
        # 应用门控
        gated_x = x * gate
        
        return gated_x


# class AdaptiveDistillationLoss(nn.Module):
#     def __init__(self, temperature=1.0):
#         super().__init__()
#         self.temperature = temperature
    
#     def forward(self, student, teacher):
#         student_norm = F.normalize(student, p=2, dim=1)
#         teacher_norm = F.normalize(teacher, p=2, dim=1)

#         attention_weights = 
        
#         cosine_similarity = torch.sum(student_norm * teacher_norm) / self.temperature

#         loss = (1 - cosine_similarity) * 


def adaptive_distillation_loss(student_features, teacher_features, attention_weights, temperature=1.0):
    """自适应知识蒸馏损失
    
    Args:
        student_features: 学生模型特征 [B, C, H, W]
        teacher_features: 教师模型特征 [B, C, H, W]
        attention_weights: 注意力权重 [B, H, W]
        temperature: 温度参数
        
    Returns:
        loss: 蒸馏损失
    """
    # 归一化特征以计算余弦相似度
    student_norm = F.normalize(student_features, p=2, dim=1)
    teacher_norm = F.normalize(teacher_features, p=2, dim=1)
    
    # 计算余弦相似度 (B, H, W)
    cosine_similarity = torch.sum(student_norm * teacher_norm, dim=1)
    
    # 应用温度参数
    cosine_similarity = cosine_similarity / temperature
    
    # 计算注意力加权的蒸馏损失
    # 注意力权重越高，对应区域的损失权重也越高
    loss = (1 - cosine_similarity) * attention_weights
    
    # 对所有空间位置求平均
    loss = loss.mean()
    
    return loss
