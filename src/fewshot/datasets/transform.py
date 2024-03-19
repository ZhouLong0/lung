import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


import torch
from PIL import ImageEnhance, Image

transformtypedict = dict(Brightness=ImageEnhance.Brightness,
                         Contrast=ImageEnhance.Contrast,
                         Sharpness=ImageEnhance.Sharpness,
                         Color=ImageEnhance.Color)


class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))
        print(out)
        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')
        print(out)
        return out
    

class ImageColorThreshold(object):
    def __init__(self):
        self.transforms = []

    def __call__(self, img):
        out = img.convert('HSV')
        multiBands = out.split()
        Hband = multiBands[0]
        Sband = multiBands[1]
        Vband = multiBands[2]
        Vband = Vband.point(self.pixelProcV)
        Sband = Sband.point(self.pixelProcS)
        Hband = Hband.point(self.pixelProcH)
        out = Image.merge('HSV', (Hband, Sband, Vband))
        out = out.convert('RGB')
        return out
    
    def pixelProcH(self, x):
        val_min = int(0.5 * 255)
        val_max = int(0.65 * 255)
        if x < val_min:
            return val_min
        elif x > val_max:
            return val_max
        else:
            return x
        
    def pixelProcS(self, x):
        val_min = int(0.1 * 255)
        if x < val_min:
            return val_min
        else:
            return x

    def pixelProcV(self, x):
        val_min = int(0.5 * 255)
        val_max = int(0.9 * 255)
        if x < val_min:
            return val_min
        elif x > val_max:
            return val_max
        else:
            return x

def without_augment(size=512, enlarge=False, basic_normalisation=False, grayscale=False):
    if enlarge:
        resize = size

    if basic_normalisation:
        return transforms.Compose([
                    transforms.Resize(resize),
                    transforms.ToTensor(),
                    normalize,
                ])
    elif grayscale:
        return transforms.Compose([
                    transforms.Resize(resize),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
                    transforms.Resize(resize),
                    # ImageColorThreshold(),
                    transforms.ToTensor(),
                ])


def with_augment(size=84, disable_random_resize=False, jitter=True):
    if disable_random_resize:
        return transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        if jitter:
            return transforms.Compose([
                transforms.RandomResizedCrop(size),
                ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])