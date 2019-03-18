import cv2
import numpy as np


def detect(img):
    #1.convert to grayscale 
    copy = img
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #morphology transfer
    dilation = preprocess(gray)
    #searching for text
    region = findTextRegion(copy,dilation)
def preprocess(gray):
    #1.sobel
    sobel = cv2.Sobel(gray,cv2.CV_8U,1,0,ksize = 3)
    #2.binary img
    ret,binary = cv2.threshold(sobel,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    #dilation and erosion
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
    dilation = cv2.dilate(binary,element2,iterations = 1)
    erosion = cv2.erode(dilation,element1,iterations = 1)
    dilation2 = cv2.dilate(erosion,element2,iterations = 4)
    cv2.imwrite('binary.png',binary)
    cv2.imwrite('diation.png',dilation)
    cv2.imwrite('erosion.png',erosion)
    cv2.imwrite('diation2.png',dilation)
    return dilation2

def findTextRegion(copy,img):
    region = []
    binary,contours,hierachy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for i in contours:
        x,y,w,h = cv2.boundingRect(i)
        cv2.rectangle(copy,(x,y),(x+w,y+h),(0,255,0))
    cv2.imshow('boundary',copy)
    cv2.imwrite('contours.png',copy)
    cv2.waitKey(0)
img = cv2.imread('q2.png')
detect(img)   
















