#!/usr/bin/env python3 
# Shebang line" specifies the interpreter. 

# imports --------------------
import cv2
import numpy as np

def showMask(window_name, image):
    image_to_show = image.astype(np.uint8)*255
    cv2.imshow(window_name, image_to_show)

# Main function
def main(): # this is our main function
    print("SAVI exercise")

    # relative path
    image = cv2.imread("images/cenario.jpg")
    template = cv2.imread("images/modelo.png")
    HEIGHT,WIDTH,NC = image.shape
    height,width,num_channels = template.shape

    # Experiment of reducing template size
    # template = cv2.resize(template, (round(height/2),  round(width/2)))
    # height,width,num_channels = template.shape

    cv2.imshow('Image', image)
    cv2.imshow('Template', template)

    # Template matching for detection
    res = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED)

    res_uint8 = (res*255).astype(np.uint8)
    cv2.imshow('Matching Result', res_uint8)

    # Foind the maximum value of correlation and its row col coordinates
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    print('max_loc = ', str(max_loc))
 
    # How to extract the bbox coords
    x = max_loc[0]
    y = max_loc[1]
    w = width
    h = height

    image_annotated = image.copy()
    cv2.rectangle(image_annotated, (x, y), (x+w, y+h), (255, 0, 255), 2)
    cv2.imshow('image annotated', image_annotated)
    

    # Limitations of the template matching
    # 1. the computation demand is very large. Testing all hyphoteis
    # 2. what happens if the template as a different size in comparison to the object that appear on the model?


    # -------------------------------------------------------
    # Ex 4d) 
    # -------------------------------------------------------

    # how to get a grey image?
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Only has a single channel
    # However w need a three channel RGB image to display the templare area in color

    image_all_gray = cv2.merge([gray, gray, gray])

    cv2.imshow('image_all_gray', image_all_gray)


    # Image a selected region copy values from original image
    image_with_object_colored = image_all_gray.copy()
    image_with_object_colored[y:y+h, x:x+w, :] = image[y:y+h, x:x+w, :]

    # Make the dog more redish
    image_with_object_colored[y:y+h, x:x+w, 2] = image[y:y+h, x:x+w, 2]*1.4


    cv2.imshow('image_with_object_colored', image_with_object_colored)

    # -------------------------------------------------------
    # Ex 4c) 
    # -------------------------------------------------------

    image_2 = cv2.imread("images/cenario_2.jpg")
    image_new = cv2.imread("images/cenario.jpg")
    x, y, width, height = cv2.selectROI("Select Template", image_2)
    new_template = image_2[y:y+height, x:x+width, :].copy()
    cv2.imshow('New Template', new_template)

    res2 = cv2.matchTemplate(image_new, new_template, cv2.TM_CCOEFF_NORMED)

    res_uint8_new = (res2*255).astype(np.uint8)
    cv2.imshow('Matching Result 2', res_uint8_new)

    _, max_val, _, max_loc = cv2.minMaxLoc(res2)
    
    print('max_loc = ', str(max_loc))
   
    x = max_loc[0]
    y = max_loc[1]
    w = width
    h = height

    image_annotated = image_new.copy()
    cv2.rectangle(image_annotated, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow('image annotated 2', image_annotated)

    cv2.waitKey(0)

if __name__ == "__main__":
    main()




