import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller




class FitNet(Distiller):
    """FitNets: Hints for Thin Deep Nets"""

    def __init__(self, student, teacher, cfg):
        super(FitNet, self).__init__(student, teacher, cfg)
      
        self.kd_loss = nn.MSELoss()
        self.kd_loss_weight = cfg.FITNET.KD_WEIGHT
        self.hint_layer = cfg.FITNET.HINT_LAYER
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.FITNET.INPUT_SIZE
        )
        self.conv_reg = ConvReg(
            feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
        )
    
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

        f_s = self.conv_reg(kd_feature_student["feats"][self.hint_layer])

      
        kd_loss = self.kd_loss_weight * self.kd_loss(f_s, kd_feature_teacher["feats"][self.hint_layer])


        losses_dict = {
            "loss_ce": ce_loss,
            "loss_triplet": triplet_loss,
            "loss_kd": kd_loss,
        }
        return logits_student, losses_dict
    


class ConvReg(nn.Module):
    """Convolutional regression for FitNet"""
    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1+s_H-t_H, 1+s_W-t_W))
        else:
            raise NotImplemented('student size {}, teacher size {}'.format(s_H, t_H))
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)
        
def get_feat_shapes(student, teacher, input_size):
    data = torch.randn(1, 3, *(input_size[0], input_size[1]))
    # 记录原来的训练状态
    student_train = student.training
    teacher_train = teacher.training

    # 临时切 eval，避免 BN 在 batch=1 下报错
    student.eval()
    teacher.eval()

    with torch.no_grad():
        _, feat_s = student(data)
        _, feat_t = teacher(data)

    feat_s_shapes = [f.shape for f in feat_s["feats"]]
    feat_t_shapes = [f.shape for f in feat_t["feats"]]

    # 恢复原来的训练状态
    student.train(student_train)
    teacher.train(teacher_train)

    return feat_s_shapes, feat_t_shapes