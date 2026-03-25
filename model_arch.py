
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class DeepShadowModel(nn.Module):
    def __init__(self):
        super(DeepShadowModel, self).__init__()
        self.visual_backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        num_features = self.visual_backbone.classifier[1].in_features
        self.visual_backbone.classifier = nn.Identity() 
        
        self.visual_compress = nn.Sequential(nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.3))
        self.metadata_branch = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU())
        self.fusion_head = nn.Sequential(
            nn.Linear(256 + 32, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, image, metadata):
        if image.shape[1] == 1: image = image.repeat(1, 3, 1, 1)
        v_features = self.visual_compress(self.visual_backbone(image))
        m_features = self.metadata_branch(metadata)
        return self.fusion_head(torch.cat((v_features, m_features), dim=1))
    