import numpy as np
import cv2
import math

def VLMatrix(array):
    
    l1=np.cross(array[0],array[1])
    l2=np.cross(array[2],array[3])
    
    Vp1=np.cross(l1,l2)
    Vp1=Vp1/Vp1[2]
    
    l3=np.cross(array[0],array[2])
    l4=np.cross(array[1],array[3])
    
    Vp2=np.cross(l3,l4)
    Vp2=Vp2/Vp2[2]
    
    VL=np.cross(Vp1,Vp2)
    
    VL=VL/VL[2]
    
    H=np.zeros((3,3))
    H[0][0]=1
    H[1][1]=1
    H[2]=VL
    
    return H

def WeightedAverageRGBPixelValue(pt, img):
    
    x1=int(math.floor(pt[0]))
    x2=int(math.ceil(pt[0]))
    y1=int(math.floor(pt[1]))
    y2=int(math.ceil(pt[1]))
        
    Wp=1/np.linalg.norm(np.array([pt[0]-x1,pt[1]-y1]))
    Wq=1/np.linalg.norm(np.array([pt[0]-x1,pt[1]-y2]))
    Wr=1/np.linalg.norm(np.array([pt[0]-x2,pt[1]-y1]))
    Ws=1/np.linalg.norm(np.array([pt[0]-x2,pt[1]-y2]))
    
    pixel_value = (Wp*img[y1][x1] + Wq*img[y2][x1] + Wr*img[y1][x2] + Ws*img[y2][x2])/(Wp+Wq+Wr+Ws)
    
    pixel_value
    return pixel_value

def ProjectionImage(H,world_plane_img):
    
    ImgP=np.asarray([0.0,0.0,1.0])
    ImgQ=np.asarray([float(np.shape(world_plane_img)[1])-1.0,0.0,1.0])
    ImgR=np.asarray([0.0,float(np.shape(world_plane_img)[0])-1.0,1.0])
    ImgS=np.asarray([float(np.shape(world_plane_img)[1]-1.0),float(np.shape(world_plane_img)[0])-1.0,1.0])  
    
       
    WorldA=np.dot(H,ImgP)
    WorldA=WorldA/WorldA[2]
    WorldB=np.dot(H,ImgQ)
    WorldB=WorldB/WorldB[2]
    WorldC=np.dot(H,ImgR)
    WorldC=WorldC/WorldC[2]
    WorldD=np.dot(H,ImgS)
    WorldD=WorldD/WorldD[2]
    
    
    xmin = int(math.floor(min([WorldA[0],WorldB[0],WorldC[0],WorldD[0]])))
    xmax = int(math.ceil(max([WorldA[0],WorldB[0],WorldC[0],WorldD[0]])))
    ymin = int(math.floor(min([WorldA[1],WorldB[1],WorldC[1],WorldD[1]])))
    ymax = int(math.ceil(max([WorldA[1],WorldB[1],WorldC[1],WorldD[1]])))
       
    yLength=ymax-ymin
    xLength=xmax-xmin
       
    src_img=np.zeros((yLength,xLength,3))   
    Hn=np.linalg.pinv(H)
    Hn=Hn/Hn[2][2]
    
    for i in range(0,yLength):
        for j in range(0,xLength):
            tmp=np.array([j+xmin,i+ymin,1.0])
            xp=np.array(np.dot(Hn,tmp))
            xp=xp/xp[2]
            if((xp[0]>0)and(xp[0]<world_plane_img.shape[1]-1)and(xp[1]>0)and(xp[1]<world_plane_img.shape[0]-1)):
                src_img[i][j]=WeightedAverageRGBPixelValue(xp,world_plane_img)            
                    
         
    output_img = src_img
    return output_img



image1=cv2.imread("1.jpg")
image2=cv2.imread("2.jpg")

t1=[]
t2=[]

t1.append([246.0,1238.0,1.0])
t1.append([2035.0,394.0,1.0])
t1.append([224.0,1359.0,1.0])
t1.append([2049.0,622.0,1.0])

pts_image1=np.asarray(t1)

t2.append([231.0,55.0,1.0])
t2.append([337.0,71.0,1.0])
t2.append([230.0,289.0,1.0])
t2.append([337.0,281.0,1.0])

pts_image2=np.asarray(t2)


H1=VLMatrix(pts_image1)
output=ProjectionImage(H1,image1)
cv2.imwrite('1_2stepP.jpg',output)

H2=VLMatrix(pts_image2)
output=ProjectionImage(H2,image2)
cv2.imwrite('2_2stepP.jpg',output)









