import numpy as np
import csv
from tqdm import tqdm
from PIL import Image
import openslide
import os
import xml.etree.ElementTree as et
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt


def create_patchs_from_squares(slide_filepath, slide_name, xml_filepath, size=1728, patch_size=1728):
    """
    Generate patch images from squares in a WSI and its csv file.

    Args:
        slide_filepath (str): The path to the slide file.
        slide_name (str): The name of the slide.
        xml_filepath (str): The path to the XML file.
        size (int, optional): The size of the squares. Defaults to 1728.
        patch_size (int, optional): The size of the patch images. Defaults to 1728.
    Returns:
        None
    """
    slide = openslide.OpenSlide(slide_filepath)
    csv_file_name = "query_squares_" + slide_name + ".csv"
    f = open(csv_file_name, 'w')
    writer  = csv.writer(f)
    patchs_folder = "patchs_from_whole_slide_with_squares_" + slide_name 
    if not os.path.exists(patchs_folder):
        os.mkdir(patchs_folder)
    xroot = et.parse(xml_filepath).getroot()
    for child in tqdm(xroot[1][1]):
        polygon_name = child.attrib['name']
        annoatation_name, classe = polygon_name.split(' ')[0], polygon_name.split(' ')[1] 
        if classe in ['AN','AM','NT','VE','RE']:
            for i in range(len(annoatation_name)):
                if annoatation_name[i].isdigit() == False:
                    annoatation_name = annoatation_name[:i] + '_' + annoatation_name[i] + '_' + annoatation_name[i+1:]
                    break
            lx=[int(point.attrib['x']) for point in child]
            ly=[int(point.attrib['y']) for point in child]
            if len(lx)==5: # ie if polygone is square 5=>4 points of the square + the initial point
                maxx, minx = max(lx), min(lx)
                maxy, miny = max(ly), min(ly)
                cpt = 0
                for x_deb in range(minx, maxx-size, size//2):
                    for y_deb in range(miny, maxy-size, size//2):
                        patch = slide.read_region((x_deb+size//2-patch_size//2, y_deb+size//2-patch_size//2), 0, (patch_size, patch_size)).convert('RGB')
                        image_name = f'{annoatation_name}_{classe}_{cpt}_x_{x_deb}_y_{y_deb}'
                        patch.save(f'{patchs_folder}/{image_name}_res{patch_size}x{patch_size}.jpg') 
                        writer.writerow([image_name,classe])  
                        cpt += 1

    f.close()

def create_patchs_from_polygons(slide_file, xml_file, slide_code, size=1728):
    """
    Generates patches from a WSI based on the polygons defined in an XML file and its csv file.
    
    Parameters:
        slide_file (str): The path to the whole slide image file.
        xml_file (str): The path to the XML file containing the polygon definitions.
        slide_code (str): The code associated with the slide.
        size (int, optional): The size of each patch image. Defaults to 1728.
    Returns:
        None
    """
    slide_name=xml_file.split('_')[0]
    f = open("query_WSI_" + slide_name + ".csv", 'w')
    writer  = csv.writer(f)
    patchs_folder = "patchs_from_whole_slide_with_polygons_" + slide_name 
    if not os.path.exists(patchs_folder):
        os.mkdir(patchs_folder)

    xroot = et.parse(xml_file).getroot() 
    all_polygons = {}
    for child in xroot[1][1]:
        poly_name = child.attrib['name']
        for i in range(len(poly_name)):
            if poly_name[i]==' ' or poly_name[i].isdigit():
                poly_name = poly_name[:i]
                break
        points=[(int(point.attrib['x']),int(point.attrib['y'])) for point in child]
        all_polygons[poly_name] = Polygon(points)

    slide = openslide.OpenSlide(slide_file)
    slide_w, slide_h = slide.level_dimensions[0]
    for i in tqdm(range(2*slide_w//size)):
        for j in range(2*slide_h//size):
            point = Point(i*size//2 + size//2, j*size//2 + size//2)
            for poly in all_polygons:
                if all_polygons[poly].contains(point):
                    classe = poly
                    patch = np.array(slide.read_region((i*size//2, j*size//2), 0, (size, size)).convert('RGB'))
                    frac_blanc = (patch[:,:,:3].sum(2)>3*240).sum()/(size*size)
                    if frac_blanc<0.5:
                        image_name = f'{slide_code}_x_{i}_y_{j}'
                        Image.fromarray(patch).save(f'{patchs_folder}/{image_name}_res{size}x{size}.jpg') 
                        writer.writerow([image_name,classe])  
    f.close()

def create_patchs_whole_slide(slide_filepath, slide_name, size=1728):
    """
    Create patches from a random WSI (no annoations).

    Parameters:
    - slide_filepath (str): The filepath of the whole slide image.
    - slide_name (str): The name of the slide.
    Returns:
    None
    """
    slide = openslide.OpenSlide(slide_filepath)
    slide_w, slide_h = slide.level_dimensions[0]
    csv_file_name = "query_UNKNOWN_WSI_" + slide_name + ".csv"
    f = open(csv_file_name, 'w')
    writer  = csv.writer(f)

    patchs_folder = "patchs_from_whole_slide_" + slide_name 
    if not os.path.exists(patchs_folder):
        os.mkdir(patchs_folder)

    for i in tqdm(range(slide_w//(size//2))):
        for j in range(slide_h//(size//2)):
            patch = np.array(slide.read_region((i*(size//2), j*(size//2)), 0, (size, size)).convert('RGB'), dtype='uint8')
            if np.mean(patch)==0.0:
                pass
            else:           
                frac_blanc = (patch.sum(2)>3*240).sum()/(size*size) # 240 IS THE THRESHOLD
                if frac_blanc<0.5:
                    im2save = Image.fromarray(patch)
                    image_name = f'{slide_name.split("_")[0][:-1]}_{slide_name.split("_")[0][-1:]}_x_{i}_y_{j}' #exp 63A => 63_A_x_0_y_0 with classe = UNKNOWN
                    im2save.save(f'{patchs_folder}/{image_name}_res{size}x{size}.jpg') 
                    writer.writerow([image_name,'UNKNOWN'])
    f.close()


"""usage examples """

# create_patchs_from_squares("36H/36H_Wholeslide_Default_Extended.tif", "36H", "36H_Annotations.xml")

# create_patchs_from_polygons("36I/36I_Wholeslide_Default_Extended.tif", "36I_Annotations.xml", "36_I")

# create_patchs_from_whole_slide("63A/63A_Wholeslide_Default_Extended.tif", "63A_WSI")