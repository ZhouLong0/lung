from PIL import Image
from torch.utils.data import Dataset

class LungSet(Dataset):
    def __init__(self, dataframe, data_path, transform = None):
        '''
        Dataset for the lung dataset
        
        Args:
            dataframe: dataframe with {image_name, label}
            data_path: external directory where the images of all classes are
            transform: transform to be applied to the image
        '''
        self.df=dataframe
        self.data_path = data_path
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.data_path+'/'+self.df['classe'].iloc[idx]+'/'+self.df['patch'].iloc[idx]               
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.df['label'].iloc[idx]
    