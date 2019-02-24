import math
import os

import cv2
import numpy as np

DIR_NAME = 'milk/'

for folder in ['High', 'Low', 'Medium', 'No adulterant']:
    print(folder)

    for img_name in os.listdir(DIR_NAME + folder):
        img = cv2.imread(DIR_NAME + folder + '/' + img_name, 0)
        cimg = cv2.imread(DIR_NAME + folder + '/' + img_name)
        img = cv2.resize(img, None, fx=0.2, fy=0.2)
        cimg = cv2.resize(cimg, None, fx=0.2, fy=0.2)

        ret, gray = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
        cv2.imshow('gray_otsu', gray)
        kernel = np.ones((3, 3), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=5)

        # Find contour of rectangle
        _, contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        points = None
        cimg2 = cimg.copy()
        for c in contours:
            area = (cv2.contourArea(c))
            if area > 100000 and area < 180000:
                # Crop image points
                points = cv2.boundingRect(c)
                cv2.drawContours(cimg2, c, -1, (255, 255, 0))
        if points:
            # Crop image to proceed
            x1, y1, w, h = points
            cimg = cimg[y1:y1 + h, x1:x1 + w]
            img = img[y1:y1 + h, x1:x1 + w]

        cv2.imshow('out', cimg2)

        r = cimg[ : , : , 0]
        g = cimg[ : , : , 1]
        b = cimg[ : , : , 2]

        ret, otsu = cv2.threshold(b, 170, 255, cv2.THRESH_BINARY)
        cv2.imshow('b_otsu', otsu)

        # Opening
        kernel = np.ones((3, 3), np.uint8)
        otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel, iterations=5)

        _, contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        means = np.zeros((6, 4))
        count = 0
        for c in contours:
            center, radius = cv2.minEnclosingCircle(c)
            area = cv2.contourArea(c)

            if (radius > 20 and radius < 40 and abs(area - math.pi * radius * radius) < 1200):
                cv2.drawContours(cimg, c, -1, (0, 255, 0), cv2.FILLED)
                mask = np.zeros((cimg.shape[0], cimg.shape[1], 1), dtype=np.uint8)
                cv2.fillPoly(mask, [c], 255)
                cv2.circle(cimg, (int(center[0]), int(center[1])), int(radius), (244, 0, 0))
                res = cv2.bitwise_and(cimg, cimg, mask=mask)
                cv2.imshow('mask', res)
                cv2.imshow('m', mask)

                mean_val = cv2.mean(cimg, mask=mask)
                if count >= 6:
                    print("More than 6 contours found")
                    cv2.waitKey(0)
                    break
                means[count, :] = mean_val

                count += 1

        print(','.join([str(i) for i in list(np.mean(means, axis=0)[:])]))
        cv2.imshow("cimg", cimg)

        if (count != 6):
            print("Circles found", count)
            cv2.waitKey(0)
            pass
