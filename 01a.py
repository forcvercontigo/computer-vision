# -*- coding: utf-8 -*-

import numpy as np
import cv2
#  detect vertical line
def VerticalLineDetect(origin,gray):
     # Canny edge detection
      edges = cv2.Canny(gray, 30, 240)
      minLineLength = 1
      maxLineGap = 100
      l = []
      l = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
      if l.any():
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap).tolist()
      else:
            return False
      sorted_lines = sorted(lines, key=lambda x: x[0])
      # 纵向直线列表
      vertical_lines = []
      for line in sorted_lines:
            for x1, y1, x2, y2 in line:
                # 在图片上绘制纵向直线
                if abs(x1 - x2)<5:
                        print(line)
                        vertical_lines.append((x1, y1, x2, y2))
                        cv2.line(origin, (x1, y1), (x2, y2), (0, 255, 0), 10)
      if vertical_lines:
            return True
      else:
            return False
def image_preprocess(image):
      # convert to grayscale 
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      # convert to binaryscale
      _, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
      # dilation
      kernel = np.ones((3, 3), np.uint8)
      dilated = cv2.dilate(threshold, kernel, iterations=1)    
      return dilated
def findtable(origin, image):
      # detect contours
      _,contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      if not contours:
            return False
      # find the max contours
      tree = hierarchy[0]
      max_size = 0
      outer_contour_index = None
      for i, contour in enumerate(contours):
          size = cv2.contourArea(contour)
          if size > max_size:
              max_size = size
              outer_contour_index = i
      contour = contours[outer_contour_index]
      #find the rectangle to cover the table
      coordinates_x = []
      coordinates_y = []
      for coordinates1 in contour:
            for coordinates in coordinates1:
                  tempx = coordinates[0]
                  tempy = coordinates[1]
                  coordinates_x.append(tempx)
                  coordinates_y.append(tempy)
      minx = min(coordinates_x)
      miny = min(coordinates_y)
      maxx = max(coordinates_x)
      maxy = max(coordinates_y)
      newImage = origin.copy()
      table = newImage[miny:maxy, minx:maxx]
      print('table')
      cv2.imwrite('cut.png', table)
      cut = cv2.imread('cut.png')
      cv2.imshow('cut',cut)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
      #set the table area to 255, in order to find next table
      table[:] = 255 
      cv2.imshow('newImage', newImage)
      cv2.imwrite('new.png', newImage)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
      return True

new_img = cv2.imread('q1.png',0)
cv2.imwrite('new.png',new_img )  
new = True
#find the table and remove it and continue to search for table
while new :
      new_img = cv2.imread('new.png')
      origin_image = new_img.copy()
      processed_image = image_preprocess(new_img)
      new = findtable(origin_image, processed_image)
      if not VerticalLineDetect(new_img.copy(),new_img):
            break

      
