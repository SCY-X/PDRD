import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller


def csd_loss(student_features, teacher_features, top_k, temp_student, temp_teacher):
    # Normalize features with L2 normalization
    student_features = F.normalize(student_features, p=2, dim=1)
    teacher_features = F.normalize(teacher_features, p=2, dim=1)

    # Compute similarity matrix for teacher features
    teacher_similarity = teacher_features.double().mm(teacher_features.double().t())
    # Compute cross-similarity matrix between student and teacher features
    cross_similarity = student_features.double().mm(teacher_features.double().t())

    # Get the top-k indices of teacher similarity for retrieval
    teacher_topk_values, teacher_topk_indices = torch.sort(teacher_similarity, dim=1, descending=True)
    topk_gallery_indices = teacher_topk_indices[:, :top_k]

    # Gather top-k similarity scores for student predictions
    student_topk_scores = torch.gather(cross_similarity, 1, topk_gallery_indices)

    # Extract top-k similarity scores from teacher similarity matrix
    teacher_topk_scores = teacher_topk_values[:, :top_k]

    # Compute probability distributions for student and teacher
    student_probs = F.log_softmax(student_topk_scores / temp_student, dim=1)
    teacher_probs = F.softmax(teacher_topk_scores / temp_teacher, dim=1)

    # Compute KL divergence loss
    loss = F.kl_div(student_probs, teacher_probs.detach(), reduction='sum') / teacher_features.shape[0]

    return loss


class CSD(Distiller):
    """Contextual Similarity Distillation for Asymmetric Image Restrieval. CVPR2023"""

    """
    The original paper adopts an offline approach to extracting gallery features from the entire training set. 
    However, through practical experiments, it was found that directly extracting gallery features for each batch achieves better performance. 
    Therefore, in this project, I chose to use the latter approach. It is important to note that this modification is based on 
    personal experimentation and is not directly related to the original paper. The effectiveness of this approach may vary 
    depending on the specific dataset or experimental settings.
    """

    def __init__(self, student, teacher, cfg):
        super(CSD, self).__init__(student, teacher, cfg)

        self.k = cfg.CSD.TOPK
        self.tq = cfg.CSD.TEMPERATURE_QUERY
        self.tg = cfg.CSD.TEMPERATURE_GALLERY
        self.kd_loss_weight = cfg.CSD.KD_WEIGHT

        self.t_dim = cfg.CSD.T_DIM
        self.s_dim = cfg.CSD.S_DIM
        self.conv_reg = ConvReg(cfg.CSD.S_DIM, cfg.CSD.T_DIM)
      
    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.conv_reg.named_parameters())
    
    def get_train_extra_parameters(self):
        num_p = 0
        for p in self.conv_reg.parameters():
            num_p += p.numel()
        return num_p / 1e6
    
    def forward_train(self, image, kd_image, target, kd_target, **kwargs):

        logits_student, feature_student = self.student(image)
        ce_loss = self.ce_loss_weight * self.ce_loss(logits_student, target)
        triplet_loss = self.tri_loss_weight * self.triplet_loss(feature_student["pooled_feat"], target)

        _, kd_feature_student = self.student(kd_image)
        _, kd_feature_teacher = self.teacher(kd_image)
        
        fs = F.adaptive_avg_pool2d(self.conv_reg(kd_feature_student["feats"][-1]), 1).flatten(1)
        ft = F.adaptive_avg_pool2d(kd_feature_teacher["feats"][-1], 1).flatten(1)

        kd_loss = self.kd_loss_weight * csd_loss(fs, ft, self.k, self.tq, self.tg)

        losses_dict = {
            "loss_ce": ce_loss,
            "loss_triplet": triplet_loss,
            "loss_kd": kd_loss,
        }
        return logits_student, losses_dict



class ConvReg(nn.Module):
    """Convolutional regression for FitNet"""
    def __init__(self, s_C, t_C, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)
 