import torch
import torch.nn as nn
import sys
sys.path.append('/home/geng_liu/CL/dinov2/dinov2')
from dinov2.hub.backbones import dinov2_vitb14_reg
# 需要包装dino来return一个dict

class DINO(nn.Module):
    def __init__(self):
        super(self).__init__()
        model_path = "/home/geng_liu/CL/pkls/dino/dinov2_vitb14_reg4_pretrain.pth"
        self.dino_model = dinov2_vitb14_reg(pretrained=False)
        self.dino_model.load_state_dict(torch.load(model_path), strict=False)

    def _forward_impl(self, x):
        features = self.dino_model(x)  # [bs, 512]
        # x = self.fc(x)
        return {
            'fmaps': [], # 不支持feature map
            'features': features
        }

    def forward(self, x):
        return self._forward_impl(x)
    

def dino(pretrained=False, progress=True, **kwargs):
    model = DINO()
    return model