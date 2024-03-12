import os
import PIL
import argparse
import tifffile as tif 
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import openslide
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import torch.nn as nn
from histomicstk.preprocessing.color_normalization import reinhard
import warnings
warnings.filterwarnings("ignore")

SEED = 5
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore

av = torch.cuda.is_available()
print("CUDA Available :", av)
if av:
    print("Number of available CUDA :", torch.cuda.device_count())
    print("CUDA Name : ", torch.cuda.get_device_name(0))
    print("Current used CUDA :", torch.cuda.current_device())
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Running on", device, "Done.")
from utils import *

def main():
    parser = argparse.ArgumentParser(description='Generate predictions for WSI.')
    parser.add_argument('--slide_path', type=str, default="/home/aymen/Desktop/AymenRouen/lung_cancer_Rouen/WSIs/", help='path to WSI')
    parser.add_argument('--slide_name', type=str, default='1_A', help='slide name')
    parser.add_argument('--vis_scale', type=float, default=0.01, help='visual_scale')
    parser.add_argument('--patch_size', type=int, default=1000, help='patch size')
    parser.add_argument('--overlap', type=int, default=500, help='overlapping')
    parser.add_argument('--perc_wpx', type=float, default=0.8, help='percentage of white pixels')
    parser.add_argument('--perc_bpx', type=float, default=0.1, help='percentage of black pixels')
    parser.add_argument('--enlarge', type=int, default=20, help='enlargement')
    
    # Parse command-line arguments
    args = parser.parse_args()
    slide_name=args.slide_name
    slide_path=args.slide_path
    vis_scale=args.vis_scale
    patch_size=args.patch_size
    overlap=args.overlap
    perc_wpx=args.perc_wpx
    perc_bpx=args.perc_bpx
    step=args.overlap
    enlarge=args.enlarge
    real_enlarge = int(enlarge/vis_scale)
    if os.path.exists("patches_"+slide_name)==False:
        os.mkdir("patches_"+slide_name)
    print('slide_path :', slide_path)
    print('slide_name : ', slide_name)
    slide = openslide.open_slide(slide_path+slide_name+'.mrxs')
    slide_w, slide_h = slide.dimensions
    print(slide_name, ",slide dimensions :", slide_w, slide_h)
    W, H = int(slide_w * vis_scale), int(slide_h * vis_scale)
    array_slide = np.array(slide.get_thumbnail((W, H)).convert("RGB"))

    xywh = detect_tissue_regions(array_slide)
    real_x, real_y, real_w, real_h = np.array(xywh // vis_scale, np.int64)
    array_slide = np.array(slide.get_thumbnail((W, H)).convert("RGB"))
    x, y, width, height = xywh
    print('No enlargement')
    print(f"x={x}, y={y}, width={width}, height={height}")
    print(f"coordinates real_x={real_x}, real_y={real_y}, real_width={real_w}, real_height={real_h}")
    print('with enlargement')
    x_start,y_start,x_end,y_end = x-enlarge, y-enlarge, x+enlarge, y+enlarge
    real_x, real_y, real_w, real_h = real_x-real_enlarge, real_y-real_enlarge, real_w+real_enlarge, real_h+real_enlarge
    print(f"x_start={x_start}, y_start={y_start}, x_end={x_end}, y_end={y_end}")
    print(f"coordinates real_x={real_x}, real_y={real_y}, real_width={real_w}, real_height={real_h}")
    scaled_slide = np.array(array_slide)[y_start : y_end + height, x_start : x_end + width]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    ax1.imshow(array_slide); ax1.axis("off")  # show_thumbnail
    ax1.add_patch(Rectangle(xy=xywh[:2], width=xywh[-2], height=xywh[-1], edgecolor="k", facecolor="none", lw=1.5))  # type: ignore
    ax2.imshow(scaled_slide); ax2.axis("off")
    plt.tight_layout()
    plt.savefig("generated_WSIs/"+slide_name.split('.')[0]+f"_AllWSI&CloseWSI.jpeg",bbox_inches='tight', pad_inches=0.2, dpi=150)
    
    args={"slide_name": slide_name.split('.')[0], "patches_path":f"patches_{slide_name.split('.')[0]}/"}

    coords, all_coords_x, all_coords_y, all_patchs = gen_tiles(slide,real_x,real_y,real_w,real_h, 
                                                                        patch_size, patch_size, step,
                                                                        args, perc_wpx, perc_bpx, save=False)
    
    image_array = gen_image(width,height,all_coords_x, all_coords_y, fill=int(1000*vis_scale))
    slide_weighted = cv2.addWeighted(src1=scaled_slide[enlarge:-enlarge, enlarge:-enlarge], src2=image_array, alpha=0.8, beta=0.2, gamma=0.1)

    im = Image.fromarray(slide_weighted) # float32
    im.save("generated_WSIs/"+f"{slide_name.split('.')[0]}_slide_weighted.tif", "TIFF")

    plt.figure(figsize=(16,16))
    plt.imshow(slide_weighted)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("generated_WSIs/"+slide_name.split('.')[0]+f"_WeightedWSI.jpeg",bbox_inches='tight', pad_inches=0.2, dpi=150)
    print('Done!')
if __name__ == "__main__":
    main()