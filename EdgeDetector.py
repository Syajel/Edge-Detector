import matplotlib.pyplot as plt
import math
import numpy as np
from numpy import asarray
from PIL import Image

def CalculateIntegral(imgArray):
    integralImgArray = np.copy(imgArray)
    integralImgArray = integralImgArray.astype("uint64")
    for i in range (0,integralImgArray.shape[0]):
        for j in range (0,integralImgArray.shape[1]):
            if i == 0 and i != j :
                val = imgArray[i,j]+integralImgArray[i,j-1]
            elif j == 0 and i != j :
                val = imgArray[i,j]+integralImgArray[i-1,j]
            elif i == 0 and j == 0 :
               val = imgArray[0,0]              
            else:
                val = imgArray[i,j]+integralImgArray[i-1,j]+integralImgArray[i,j-1]-integralImgArray[i-1,j-1]
            integralImgArray[i,j] = val
    return integralImgArray

def CalculateLocalSum(integralImgArray,p0,p1):
    if p0[0] <= 0 and p0[1] <= 0:
        localSum = integralImgArray[p1[0],p1[1]]
    elif p0[0] == 0:
        localSum = integralImgArray[p1[0],p1[1]] - integralImgArray[p1[0],p0[1]-1]
    elif p0[1] == 0:
        localSum = integralImgArray[p1[0],p1[1]] - integralImgArray[p0[0]-1,p1[1]] 
    else:
        localSum = integralImgArray[p0[0]-1,p0[1]-1] + integralImgArray[p1[0],p1[1]] - integralImgArray[p0[0]-1,p1[1]] - integralImgArray[p1[0],p0[1]-1]
    return localSum

def EdgeDetect(integralImgArray,kernelSize):
    mid = (kernelSize-1)//2
    output1 = np.copy(integralImgArray)
    output1 = output1.astype("uint64")
    output2 = np.copy(integralImgArray)
    output2 = output2.astype("uint64")
    for i in range (mid,integralImgArray.shape[0]-mid):
        for j in range (mid,integralImgArray.shape[1]-mid):
           
            h1pos = CalculateLocalSum(integralImgArray, p0=[i - mid, j - mid], p1=[i + mid,j -1])
            h1neg = -1 *  CalculateLocalSum(integralImgArray, p0=[i - mid,j + 1], p1=[i + mid,j + mid]) 
      
            h2pos = CalculateLocalSum(integralImgArray, p0=[i + 1, j - mid], p1=[i + mid,j + mid])
            h2neg = -1 *  CalculateLocalSum(integralImgArray, (j - mid, i - mid), (i - 1,j + mid))
            output1[i][j] = math.sqrt((h1pos + h1neg)**2 + (h2pos + h2neg)**2)
            midCellVal = CalculateLocalSum(integralImgArray,(i,j),(i,j))
            sumNoMid = CalculateLocalSum(integralImgArray,(i-mid,j-mid),(i+mid,j+mid)) - midCellVal
            g1 = -1 * sumNoMid
            g2 = ((kernelSize*kernelSize)-1) * midCellVal
            g = int(g1 + g2)
            output2[i,j] = g
    return output1, output2



def RefineEdge(integralEdgeArray,kernelSize,ratio):
    mid = (kernelSize-1)/2
    output = np.copy(integralEdgeArray)
    output = output.astype("uint64")
    for i in range (0,integralEdgeArray.shape[0]):
        for j in range (0,integralEdgeArray.shape[1]):
            if i - mid < 0:
                minxIndex = 0
            else:
                minxIndex = int(i-mid)
            if i + mid > integralEdgeArray.shape[0]-1:
                maxxIndex = int(integralEdgeArray.shape[0]-1)
            else:
                maxxIndex = int(i+mid)
            if j - mid < 0:
                minyIndex = 0
            else:
                minyIndex = int(j-mid)
            if j + mid > integralEdgeArray.shape[1]-1:
                maxyIndex = int(integralEdgeArray.shape[1]-1)
            else:
                maxyIndex = int(j+mid)
            mean = CalculateLocalSum(integralEdgeArray,(minxIndex,minyIndex),(maxxIndex,maxyIndex)) / (kernelSize*kernelSize)
            integralVal = CalculateLocalSum(integralEdgeArray,(i,j),(i,j))
            if integralVal > mean*ratio:
                output[i,j] = integralVal
            else:
                output[i,j] = 0


im = Image.open("L4.jpg").convert('L')
imgArr = asarray(im)
imIntegral = CalculateIntegral(imgArr)



imEdge1d, imEdge2d = EdgeDetect(imIntegral, 121)

max1 = int(np.max(imEdge1d))
max2 = int(np.max(imEdge2d))

plt.subplot(1, 2, 1)
plt.imshow(imEdge1d,cmap = "gray")

plt.subplot(1, 2, 2)
plt.imshow(imEdge2d,cmap = "gray")
