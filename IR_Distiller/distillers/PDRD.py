import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    if len(x.size()) == 2:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        dist = torch.cdist(x, y, p=2).squeeze(0)

    elif len(x.size()) == 3:
        dist = torch.cdist(x, y, p=2)
    return dist


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        print("Mish activation loaded...")

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class PDRD(Distiller):
    """
    Pairwise difference relational distillation for object re-identification. PR2024
    """

    def __init__(self, student, teacher, cfg):
        super(PDRD, self).__init__(student, teacher, cfg)
        
        self.batch_size = cfg.SOLVER.IMS_DISTILLATION_PER_BATCH
        self.mish = Mish()
        self.kd_loss_weight = cfg.PDRD.KD_WEIGHT


    def forward_train(self, image, kd_image, target, kd_target, **kwargs):

        logits_student, feature_student = self.student(image)
        ce_loss = self.ce_loss_weight * self.ce_loss(logits_student, target)
        triplet_loss = self.tri_loss_weight * self.triplet_loss(feature_student["pooled_feat"], target)

        _, kd_feature_student = self.student(kd_image)
        _, kd_feature_teacher = self.teacher(kd_image)

        f_s = F.normalize(F.adaptive_avg_pool2d(kd_feature_student["feats"][-1], 1).flatten(1), p=2, dim=1, eps=1e-12)
        f_t = F.normalize(F.adaptive_avg_pool2d(kd_feature_teacher["feats"][-1], 1).flatten(1), p=2, dim=1, eps=1e-12)
        s_dist = torch.mm(f_s.float(), f_s.float().t())
        t_dist = torch.mm(f_t.float(), f_t.float().t())

        s_sim_all = self.all_rank(s_dist)
        t_sim_all = self.all_rank(t_dist)

        kd_loss = self.kd_loss_weight * torch.mean(torch.norm((t_sim_all - s_sim_all), p=2, dim=1))

        losses_dict = {
            "loss_ce": ce_loss,
            "loss_triplet": triplet_loss,
            "loss_kd": kd_loss,
        }
        return logits_student, losses_dict
    


    def all_rank(self, dist):
        dist_repeat = dist.unsqueeze(dim=1).repeat(1, self.batch_size, 1)
        # compute the difference matrix
        dist_diff = dist_repeat - dist_repeat.permute(0, 2, 1)
        # pass through the relu
       
        dist_sg = self.mish(dist_diff).cuda().view(self.batch_size, -1)
    
        return dist_sg





