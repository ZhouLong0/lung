import cv2
import os 
import torch
import numpy as np
from histomicstk.preprocessing.color_normalization import reinhard
from tqdm import tqdm 
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings("ignore")

seed=2024
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def bpx_ratio(im, thresh=30):
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) # Convert to grayscale
    mask = np.zeros((gray.shape[0],gray.shape[1]), np.uint8)   # Create a mask that selects the pixels close to white
    black_pixels = np.sum(gray<=thresh)      # Count the number of pixels in the mask
    bpx_ratio=black_pixels/(gray.shape[0]*gray.shape[1]) # Calculate the percentage of almost white pixels
    return round(bpx_ratio,4)
        
def wpx_ratio(im, thresh=190):
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)                             # Convert the image to grayscale
    _, thresholded_image = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Threshold the grayscale image to separate white pixels
    white_pixels = np.sum(thresholded_image == 255)                         # Count the number of white pixels
    wpx_ratio=white_pixels/(gray.shape[0]*gray.shape[1])                    # Calculate the percentage of almost white pixels
    return round(wpx_ratio,4)

def detect_tissue_regions(array_slide):
    gray = cv2.cvtColor(array_slide, cv2.COLOR_RGB2GRAY)    # Convert the image to grayscale
    _,binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)# Threshold the image to create a binary mask
    output = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# Find contours in the binary mask
    contours = output[0] if len(output)==2 else output[1]
    largest_contour = max(contours, key=cv2.contourArea)# Find the contour with the largest area (largest tissue region)
    x, y, w, h = cv2.boundingRect(largest_contour) # Get the coordinates of the bounding rectangle around the largest tissue region
    return np.array([x,y,w,h])

def gen_image(width,height,all_coords_x, all_coords_y, fill=5):
    image_size=(width,height)
    background_color=(255, 255, 255)
    image_array = np.full((image_size[1], image_size[0], 3), background_color, dtype=np.uint8)
    x_scaled = ((np.array(all_coords_x) - min(all_coords_x)) / (max(all_coords_x) - min(all_coords_x))) * (image_size[0] - 1)
    y_scaled = ((np.array(all_coords_y) - min(all_coords_y)) / (max(all_coords_y) - min(all_coords_y))) * (image_size[1] - 1)
    x_pixels = np.round(x_scaled).astype(int)
    y_pixels = np.round(y_scaled).astype(int)
    for i,(x_p, y_p) in enumerate(zip(x_pixels, y_pixels)):
        try: 
            image_array[y_p:y_p+fill, x_p:x_p+fill] = (255,0,128)
        except: 
            print('ERROR!')
            print(image_array.shape)
    return image_array

# def get_BrightandDark_perc(pil_image, bright_threshold=199, dark_threshold=20):
#     grayscale_image = pil_image.convert('L')
#     pixels = list(grayscale_image.getdata())

#     # Count bright/white and dark/black pixels
#     bright_pixels = sum(1 for pixel in pixels if pixel > bright_threshold)
#     dark_pixels = sum(1 for pixel in pixels if pixel < dark_threshold)

#     # Calculate percentages
#     total_pixels = len(pixels)
#     bright_percentage = (bright_pixels / total_pixels) * 100
#     dark_percentage = (dark_pixels / total_pixels) * 100
#     return bright_percentage, dark_percentage

def get_BrightandDark_perc(pil_image, bright_threshold=200, dark_threshold=20):
    grayscale_image = np.array(pil_image.convert('L'))
    # Count bright/white and dark/black pixels
    bright_pixels = np.sum(grayscale_image > bright_threshold)
    dark_pixels = np.sum(grayscale_image < dark_threshold)
    # Calculate percentages
    total_pixels = grayscale_image.size
    bright_percentage = (bright_pixels / total_pixels)
    dark_percentage = (dark_pixels / total_pixels)
    return bright_percentage, dark_percentage

def gen_tiles(slide, start_x, start_y, width, height, patch_width, patch_height, step, args, perc_wpx, perc_bpx, save=False):
    slide_name=args['slide_name']
    patches_path=args['patches_path']
    if os.path.exists(patches_path)==False:
        os.mkdir(patches_path)
    coords, d={}, {}
    for x in tqdm(range(start_x, start_x+width, step)):
        cols,coords_cols=[],[]
        white=[]
        for y in range(start_y, start_y+height, step):
            # patch = np.uint8(slide.read_region((x,y), 0, (patch_width, patch_height)).convert('RGB'))
            patch = slide.read_region((x,y), 0, (patch_width, patch_height)).convert('RGB')
            wpx,bpx = get_BrightandDark_perc(patch)
            if wpx<perc_wpx and bpx<perc_bpx :
                cols.append(patch)
                coords_cols.append((x,y))
                if save: 
                    patch.save(patches_path+slide_name+"_row_"+str(x//step)+"_col_"+str(y//step)+'.jpg');
            else:
                coords_cols.append((-1,-1))
            del patch
        if cols!=[]: 
            d[x]=cols
            coords[x]=coords_cols
        del cols
    all_patchs = [patch for patches in d.values() for patch in patches]
    all_coords_x = [x for points in coords.values() for x, y in points if x != -1 and y != -1]
    all_coords_y = [y for points in coords.values() for x, y in points if x != -1 and y != -1]
    return coords, all_coords_x, all_coords_y, all_patchs

def gen_image_from_patchs(width,height,all_coords_x, all_coords_y, color=(238,130,238), fill=4):
    image_size=(width,height)
    background_color=(255, 255, 255)
    image_array = np.full((image_size[1], image_size[0], 3), background_color, dtype=np.uint8)
    # Map coordinates to pixel positions
    x_scaled = ((np.array(all_coords_x) - min(all_coords_x)) / (max(all_coords_x) - min(all_coords_x))) * (image_size[0] - 1)
    y_scaled = ((np.array(all_coords_y) - min(all_coords_y)) / (max(all_coords_y) - min(all_coords_y))) * (image_size[1] - 1)
    x_pixels, y_pixels = np.round(x_scaled).astype(int), np.round(y_scaled).astype(int) # Round to integers to get pixel coordinates
    for i,(x_p, y_p) in enumerate(zip(x_pixels, y_pixels)): # Set the color of the points on the image
        try: 
            image_array[y_p:y_p+fill, x_p:x_p+fill] = color
        except: 
            print('ERROR!')
    return image_array

def colors_generation(y_har_mean_preds, tumor=(220,20,60), poumon=(50,205,50), stroma=(255,165,0)):
    colors_har = []
    for k in range(len(y_har_mean_preds)):
        if   y_har_mean_preds[k].item()==0: colors_har.append(stroma)
        elif y_har_mean_preds[k].item()==1: colors_har.append(poumon)
        elif y_har_mean_preds[k].item()==2: colors_har.append(tumor)
    return colors_har

def gen_prediction_image(width,height,all_coords_x, all_coords_y, predicted_colors, fill=5):
    image_size=(width,height)
    background_color=(255, 255, 255)
    image_array = np.full((image_size[1], image_size[0], 3), background_color, dtype=np.uint8)
    x_scaled = ((np.array(all_coords_x) - min(all_coords_x)) / (max(all_coords_x) - min(all_coords_x))) * (image_size[0] - 1)
    y_scaled = ((np.array(all_coords_y) - min(all_coords_y)) / (max(all_coords_y) - min(all_coords_y))) * (image_size[1] - 1)
    x_pixels = np.round(x_scaled).astype(int)
    y_pixels = np.round(y_scaled).astype(int)
    for i,(x_p, y_p) in enumerate(zip(x_pixels, y_pixels)):
        try: 
            image_array[y_p:y_p+fill, x_p:x_p+fill] = predicted_colors[i]
        except: 
            print('ERROR!')
            print(image_array.shape, len(predicted_colors))
    return image_array

def gen_confidence(y_har_mean_proba):
    confidence=[y_har_mean_proba[k].max().item() for k in range(len(y_har_mean_proba))]
    return confidence

def gen_confidence_image(width,height,all_coords_x, all_coords_y, predicted_colors, fill=4):
    image_size=(width,height)
    background_color=(0)
    image_array = np.full((image_size[1], image_size[0]), background_color, np.uint8)
    x_scaled = ((np.array(all_coords_x) - min(all_coords_x)) / (max(all_coords_x) - min(all_coords_x))) * (image_size[0] - 1)
    y_scaled = ((np.array(all_coords_y) - min(all_coords_y)) / (max(all_coords_y) - min(all_coords_y))) * (image_size[1] - 1)
    x_pixels = np.ceil(x_scaled).astype(int)
    y_pixels = np.ceil(y_scaled).astype(int)
    for i,(x_p, y_p) in enumerate(zip(x_pixels, y_pixels)):
        image_array[y_p:y_p+fill,x_p:x_p+fill] = int(predicted_colors[i]*255)
    return image_array

def get_preds_probas(model, lung_loader, device):
    y_preds = torch.zeros(0, dtype=torch.long, device='cpu')
    y_probas = torch.zeros(0, dtype=torch.long, device='cpu')
    with torch.no_grad():
        for data in tqdm(lung_loader):
            output = model(data.to(device))
            probas=torch.softmax(output, dim=1)
            y_preds = torch.cat([y_preds, probas.argmax(dim=1).view(-1).cpu()])
            y_probas = torch.cat([y_probas, probas.cpu()])
            del data, output, probas
    return y_preds, y_probas

class LungSet(Dataset):
    def __init__(self, X, Y, transform):
        self.data = X
        self.labels=Y
        self.transform = transform  
    def ReinhardNorm(self,img):
        cnorm = {'mu': np.array([8.74108109, -0.12440419,  0.0444982]),'sigma': np.array([0.6135447, 0.10989545, 0.0286032])}
        return reinhard(np.array(img), target_mu=cnorm['mu'], target_sigma=cnorm['sigma'])      
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image=self.transform(self.ReinhardNorm(self.data[idx]))
        return image