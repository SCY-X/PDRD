import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller



class SP(Distiller):
    """Correlation Congruence for Knowledge Distillation, ICCV 2019.
    The authors nicely shared the code with me. I restructured their code to be 
    compatible with my running framework. Credits go to the original author"""

    def __init__(self, student, teacher, cfg):
        super(SP, self).__init__(student, teacher, cfg)

    
        self.kd_loss_weight = cfg.SP.KD_WEIGHT

    
    def forward_train(self, image, kd_image, target, kd_target, **kwargs):

        logits_student, feature_student = self.student(image)
        ce_loss = self.ce_loss_weight * self.ce_loss(logits_student, target)
        triplet_loss = self.tri_loss_weight * self.triplet_loss(feature_student["pooled_feat"], target)

        _, kd_feature_student = self.student(kd_image)
        _, kd_feature_teacher = self.teacher(kd_image)

        f_s = F.adaptive_avg_pool2d(kd_feature_student["feats"][-1], 1).flatten(1)
        f_t = F.adaptive_avg_pool2d(kd_feature_teacher["feats"][-1], 1).flatten(1)
        
     
        kd_loss = self.kd_loss_weight * self.similarity_loss(f_s, f_t)


        losses_dict = {
            "loss_ce": ce_loss,
            "loss_triplet": triplet_loss,
            "loss_kd": kd_loss,
        }
        return logits_student, losses_dict

    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = F.normalize(f_s.view(bsz, -1), p=2, dim=1)
        f_t = F.normalize(f_t.view(bsz, -1), p=2, dim=1)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        # G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        #G_t = torch.nn.functional.normalize(G_t)
       
        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss
