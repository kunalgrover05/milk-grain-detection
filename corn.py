import os
import cv2
import numpy as np


# Show labelled components
def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    cv2.imshow('labeled', labeled_img)


DIR_NAME = 'corn_images/'
for img_name in os.listdir(DIR_NAME):
    img = cv2.imread(DIR_NAME + img_name)
    img = cv2.resize(img, None, fx=0.3, fy=0.3)

    # For finding blacked regions
    gray = img[:, :, 1]
    (t, binary) = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel, iterations=2)
    black_count = 0

    _, contours, hierarchy = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        if cv2.contourArea(c)>100 and cv2.contourArea(c)<3000:
            cv2.drawContours(img, c, -1, (0, 255, 0), 2)
            black_count += 1

    # Binary image for segmenting
    gray = img[ : , : , 2]
    (t, binary) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    binary = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel, iterations=2)

    # Dilating our binary image to find out number of grains. We might want to merge based on Distance transform below
    sure_bg = cv2.dilate(binary,kernel,iterations=10)
    dist_transform = cv2.distanceTransform(binary,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform, 20,255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg,sure_fg)

    ########## Marker labelling ##############
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    markers[unknown==255] = 0

    grain_count = np.max(markers) - 1 # Starting from 2-final marker (0=Unknown, 1=Background)
    # Write label on each marker
    for i in range(2, grain_count):
        marker_mask = np.zeros(gray.shape, dtype="uint8")
        marker_mask[markers == i] = 255

        # Find center of mask
        moment = cv2.moments(marker_mask)
        center = (moment['m10']/moment['m00'], moment['m01']/moment['m00'])
        cv2.putText(img, "#{}".format(i), (int(center[0]) - 10, int(center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

    imshow_components(markers)

    cv2.imshow("Final", img)
    print(img_name, "Black-" + str(black_count), "Total-" + str(grain_count))
    cv2.waitKey(0)
