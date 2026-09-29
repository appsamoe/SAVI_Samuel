#!/usr/bin/env python3 
# Shebang line" specifies the interpreter. 

# imports --------------------
import cv2

# Main function
def main(): # this is our main function
    print("SAVI exercise")

    image = cv2.imread("./images/3.png", cv2.IMREAD_UNCHANGED)

    #channel game
    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]
    alpha = image[:, :, 3]
    # cv2.imshow("Blue channel", blue)
    # cv2.imshow("Green channel", green)
    # cv2.imshow("Red channel", red)
    # cv2.imshow("Alpha channel", alpha)
    cv2.imshow("normal", image)

    # edited_image = image.copy()
    # edited_image[:, :, 0] = 0
    # cv2.imshow("no blue", edited_image)
    # edited_image = image.copy()
    # edited_image[:, :, 0] = 255
    # cv2.imshow("all blue", edited_image)
    # edited_image = image.copy()
    # edited_image[:, :, 1] = 0
    # cv2.imshow("no green", edited_image)
    # edited_image = image.copy()
    # edited_image[:, :, 1] = 255
    # cv2.imshow("all green", edited_image)
    # edited_image = image.copy()
    # edited_image[:, :, 2] = 0
    # cv2.imshow("no red", edited_image)
    # edited_image = image.copy()
    # edited_image[:, :, 2] = 255
    # cv2.imshow("all red", edited_image)
    edited_image = image.copy()
    edited_image[:, :, 3] = 0
    cv2.imshow("no alpha", edited_image)
    edited_image = image.copy()
    edited_image[:, :, 3] = 255
    cv2.imshow("all alpha", edited_image)

    cv2.waitKey(0)

    






if __name__ == "__main__":
    main()