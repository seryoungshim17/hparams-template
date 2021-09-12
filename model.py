from torch import nn
import torchvision

class Resnet18Model(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        ### 모델 Layer 정의 ###
        self.model = torchvision.models.resnet18(pretrained=True)  # resnet 18 pretrained model 사용
        self.model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True) # 마지막 Layer 변경
        
        self.num_classes = num_classes
        
    def forward(self, x):
        ### 모델 structure ###
        return self.model(x)