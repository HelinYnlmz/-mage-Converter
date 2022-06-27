from math import log10, sqrt
import cv2
import numpy as np

def PSNR(original, converted):
    mse = np.mean((original - converted) ** 2)
    if(mse == 0):  
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

img = cv2.imread('mandrill.ppm', 1)
hsi =  cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

cv2.imwrite('rgb.png', img)
cv2.imwrite('hsi.png', hsi)
cv2.imwrite('ycrcb.png', ycrcb)

cv2.imshow('RGB Image',img)
cv2.imshow('HSI Image',hsi)
cv2.imshow('YCrCb Image',ycrcb)


HSIvalue = PSNR(img, hsi)
print(f"PSNR value between RGB and HSI is {HSIvalue} dB")

YCrCbvalue = PSNR(img, ycrcb)
print(f"PSNR value between RGB and YrbCb is {YCrCbvalue} dB")


cv2.waitKey(0)
cv2.destroyAllWindows()