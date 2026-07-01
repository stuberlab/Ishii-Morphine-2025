import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgba
import cv2
import tifffile
# 1. Automatic intensity adjustment
def adjust_intensity(image):
    min_val, max_val = np.min(image), np.max(image)
    return (image - min_val) / (max_val - min_val)

# 2. Convert base_image to color image using coolwarm colormap
def convert_to_coolwarm(image, cmin, cmax):
    norm = Normalize(vmin=cmin, vmax=cmax)
    colormap = plt.cm.coolwarm
    return colormap(norm(image))
def convert_to_cmap(image, cmin=-2, cmax=2, colormap=plt.cm.coolwarm):
    """
    Convert a grayscale image to an RGB image using a colormap.
    Returns an RGB image (without the alpha channel) with values in [0,1].
    """
    norm = Normalize(vmin=cmin, vmax=cmax)
    return colormap(norm(image))
# 3. Maximize intensity of overlap_image and convert to color
def convert_overlap_image(image):
    norm_image = adjust_intensity(image)
    colormap = plt.cm.Greys  # Reverse grayscale; large values -> black
    return colormap(image)

# 4. Overlay the images
def overlay_images(base_color, overlap_color, alpha=0.5):
    #print(overlap_color)
    #print(base_color.shape, overlap_color.shape)
    base_color[overlap_color == 0] = 0
    return base_color

# create a heatmap overlayed with a contour
def overlap_contour(base_image,overlap_image,cmin =  -100, cmax = 100,outputpath = False,colormap = plt.cm.coolwarm):
    # Load your data
    # Assuming `base_image` and `overlap_image` are already loaded as 3D numpy arrays.
    # Replace these with your data loading mechanism.
    #base_image = np.mean(norm_stm_array,axis = 0)
    #overlap_image = contour_img

    # Step 1: Adjust intensity of both images
    #base_image_adj = adjust_intensity(base_image)
    base_image_adj = base_image
    overlap_image_adj = adjust_intensity(overlap_image)

    # Step 2: Convert base_image to coolwarm colormap
      # Ensure symmetric range around 0
    #base_image_color = convert_to_coolwarm(base_image_adj, cmin, cmax)
    base_image_color = convert_to_cmap(base_image_adj, cmin, cmax,colormap = colormap)
    
    # Step 3: Convert overlap_image to grayscale colormap
    overlap_image_color = convert_overlap_image(overlap_image_adj)

    # Step 4: Overlay the images
    overlayed_image = overlay_images(base_image_color, overlap_image_color, alpha=1)

    # write the result
    if outputpath:
        tifffile.imwrite(outputpath, overlayed_image)
    return base_image_adj,overlayed_image


'''if __name__ == "__main__":
    # run through all
    for condition in Conditions:
        if not condition == 'Saline':
            continue
        # normalize the data

        # run the overlap script
        base_image_adj,overlayed_image = overlap_contour(np.mean(sal_array,axis = 0),contour_img,\
            cmin =  -50, cmax = 50,\
            outputpath = os.path.join(analysis_resultpath,condition + '_contour_overlap.tif'))

        # Step 5: Extract zplace = 100 slice and display the image
        zplace = 100
        plt.figure(figsize=(10, 10))
        plt.imshow(overlayed_image[zplace,imy_slice, imx_slice,])
        plt.axis('off')
        plt.title(f'{condition}_z={zplace}')
        plt.show()
'''