import torchvision

def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = torchvision.models.__dict__['resnet18'](pretrained=False)
    return model
