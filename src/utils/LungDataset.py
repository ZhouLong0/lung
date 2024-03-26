from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class LungSet(Dataset):
    def __init__(self, dataframe, data_path, transform = None, file_name = False, predictions = False):
        '''
        Dataset for the lung dataset from a dataframe
        
        Args:
            dataframe: csv split file with {image_name, label ...}
            data_path: external directory where the images of all classes are
            transform: transformation to be applied to the image

        __getitem__:
            Returns the image and the label of the idx-th image 

            Returns:
                (tensor) image: image of the idx-th image
                (int) label: label of the idx-th image
                optional: (str) file_name: name of the file
        '''
        self.df=dataframe
        self.data_path = data_path
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.file_name = file_name
        self.predictions = predictions
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        '''
        Returns the image and the label of the idx-th image 

        Returns:
            (tensor) image: image of the idx-th image
            (int) label: label of the idx-th image
        '''

        if not self.predictions:
            img_name = self.data_path+'/'+self.df['classe'].iloc[idx]+'/'+self.df['patch'].iloc[idx]               
            image = Image.open(img_name).convert('RGB')
        else:
            img_name = self.data_path+'/'+self.df['patch'].iloc[idx]
            image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)



        if self.file_name:
            return image, self.df['label'].iloc[idx], self.df['patch'].iloc[idx]

        return image, self.df['label'].iloc[idx]
    