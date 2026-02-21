import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

class CC(Distiller):
    """Correlation Congruence for Knowledge Distillation, ICCV 2019.
    The authors nicely shared the code with me. I restructured their code to be 
    compatible with my running framework. Credits go to the original author"""

    def __init__(self, student, teacher, cfg):
        super(CC, self).__init__(student, teacher, cfg)

        self.normalize = cfg.CC.NORMALIZE
        self.t_dim = cfg.CC.T_DIM
        self.s_dim = cfg.CC.S_DIM
        self.kd_loss_weight = cfg.CC.KD_WEIGHT

        self.embed_t = LinearEmbed(self.t_dim, 128)
        self.embed_s = LinearEmbed(self.s_dim, 128)


    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.embed_t.named_parameters()) + list(self.embed_s.named_parameters())
    
    def get_train_extra_parameters(self):
        num_p = 0
        for p in self.embed_t.parameters():
            num_p += p.numel()
        
        for p in self.embed_s.parameters():
            num_p += p.numel()

        return num_p / 1e6
    
    def forward_train(self, image, kd_image, target, kd_target, **kwargs):

        logits_student, feature_student = self.student(image)
        ce_loss = self.ce_loss_weight * self.ce_loss(logits_student, target)
        triplet_loss = self.tri_loss_weight * self.triplet_loss(feature_student["pooled_feat"], target)

        _, kd_feature_student = self.student(kd_image)
        _, kd_feature_teacher = self.teacher(kd_image)


        if self.normalize:
            fs = F.normalize(self.embed_s(F.adaptive_avg_pool2d(kd_feature_student["feats"][-1], 1).flatten(1)), p=2, dim=1)
            ft = F.normalize(self.embed_t(F.adaptive_avg_pool2d(kd_feature_teacher["feats"][-1], 1).flatten(1)), p=2, dim=1)
        else:
            fs = self.embed_s(F.adaptive_avg_pool2d(kd_feature_student["feats"][-1], 1).flatten(1))
            ft = self.embed_t(F.adaptive_avg_pool2d(kd_feature_teacher["feats"][-1], 1).flatten(1))

        
        delta = torch.abs(fs - ft)
     
        kd_loss = self.kd_loss_weight * torch.mean((delta[:-1] * delta[1:]).sum(1))
      
        losses_dict = {
            "loss_ce": ce_loss,
            "loss_triplet": triplet_loss,
            "loss_kd": kd_loss,
        }
        return logits_student, losses_dict



class LinearEmbed(nn.Module):
    """Linear Embedding"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(LinearEmbed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x
