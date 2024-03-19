import torch 
from torch.nn import Module
import torchvision.models as models
import torch.nn as nn
from torch.nn import Sequential

seed=2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class LungNet(Module):
    def __init__(self, model_path, num_classes):
        '''
        New transfered model from pretrained model, by changing the last layer with an adapted MLP 
        and turning off previous model's parameters gradients

        Args:
            model_path: path to the pretrained baseline model
            num_classes: number of classes to be classified

        Attributes:
            model: new transfered model using pretrained model
        '''
        super(LungNet, self).__init__()
        self.model_path = model_path
        self.num_classes = num_classes
        self.model=self.load_model()
        
    def load_model_weights(self, model, weights):
        model_dict = model.state_dict()
        weights = {k: v for k, v in weights.items() if k in model_dict}
        if weights == {}:
            print('No weight could be loaded..')
        model_dict.update(weights)
        model.load_state_dict(model_dict)

        return model
    
    def load_model(self):
        model = models.__dict__['resnet18'](pretrained=False)
        state = torch.load(self.model_path, map_location=torch.device('cpu'))
        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
        model = self.load_model_weights(model, state_dict)

        for param in model.parameters():
            param.requires_grad=False
        in_fc = model.fc.in_features
        model.fc = Sequential(nn.Flatten(),
                            nn.Linear(in_fc, 128),
                            nn.ReLU(),
                            nn.BatchNorm1d(128),
                            nn.Linear(128, 32),
                            nn.ReLU(),
                            nn.BatchNorm1d(32),
                            nn.Linear(32, self.num_classes)) 
        return model
    
    def forward(self, x):
        return self.model(x)