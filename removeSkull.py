import os
import cv2
import numpy as np


def read_process_images(folder):
        if not os.path.exists('removedSkull'):
                os.makedirs('removedSkull')
        if not os.path.exists('removedSkull/yes'):
                os.makedirs('removedSkull/yes')
        if not os.path.exists('removedSkull/no'):
                os.makedirs('removedSkull/no')
        classes = ['/yes', '/no']

        for path in classes:
                for imagename in os.listdir(folder + path):
                        img = cv2.imread(os.path.join(folder + path, imagename))
                        x = img.copy()
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                        # Thresholding the same image
                        _, thresholdedImage = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

                        # Applying Morphological Dilation
                        kernel = np.ones((15, 15), np.uint8)
                        dilated = cv2.morphologyEx(thresholdedImage, cv2.MORPH_DILATE, kernel)

                        # Getting the largest contour
                        hh, ww = img.shape[:2]
                        # get largest contour
                        contours, x = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        contours = sorted(contours, key=cv2.contourArea, reverse=True)
                        ours = contours[0]

                        # draw largest contour as white filled on black background as mask
                        mask = np.zeros((hh, ww), dtype=np.uint8)
                        cv2.drawContours(mask, [ours], 0, 255, cv2.FILLED)

                        # Applying Morphological Dilation
                        kernel = np.ones((15, 15), np.uint8)
                        eroded = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
                        eroded = cv2.morphologyEx(eroded, cv2.MORPH_ERODE, kernel)
                        eroded = cv2.morphologyEx(eroded, cv2.MORPH_ERODE, kernel)

                        # Doing and between the mask and the image
                        final = cv2.bitwise_and(img, eroded)

                      #  test = np.hstack((img, thresholdedImage, dilated, mask, eroded, final))
                      #  cv2.imwrite("test/" + imagename, test)

                        # save image
                        splitted = imagename.split('.')
                        new_name = "removedSkull" + path + '/'  + splitted[0] + "_segmented." + splitted[1]
                        cv2.imwrite(new_name, final)

                        # if i==5:
                        #      break
                        # i += 1



