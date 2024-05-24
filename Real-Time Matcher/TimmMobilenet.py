import torch
import timm
import torch.nn.functional as F


class TimmMobilenet(torch.nn.Module):
    def __init__(self, timm_name='mobilenetv3_large_100'):
        super(TimmMobilenet, self).__init__()
        self.source_model = timm.create_model(timm_name, pretrained=True)

    def forward(self, x):
        x = self.source_model.forward_features(x)
        x = self.source_model.global_pool(x)
        x = self.source_model.conv_head(x)
        x = self.source_model.act2(x)

        x = torch.flatten(x, start_dim=1)
        x = F.normalize(x, p=2, dim=1)

        return x