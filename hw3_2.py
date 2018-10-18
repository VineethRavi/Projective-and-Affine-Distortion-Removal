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

def AffineMatrix(arrays1,arrays2):
    
    ta=[]
    tb=[]
    
    l1=np.cross(arrays1[0],arrays1[1])
    m1=np.cross(arrays1[0],arrays1[2])
    l1=l1/l1[2]
    m1=m1/m1[2]
    
    l2=np.cross(arrays1[0],arrays1[3])
    m2=np.cross(arrays1[1],arrays1[2])
    l2=l2/l2[2]
    m2=m2/m2[2]
    
    ta.append([l1[0]*m1[0],l1[0]*m1[1]+l1[1]*m1[0]])
    tb.append([-l1[1]*m1[1]])
    
    ta.append([l2[0]*m2[0],l2[0]*m2[1]+l2[1]*m2[0]])
    tb.append([-l2[1]*m2[1]])
    
    A=np.asarray(ta)
    b=np.asarray(tb)
 
    tmp=np.dot(np.linalg.pinv(A),b)
    
    S=np.zeros((2,2))
    S[0][0]=tmp[0]
    S[0][1]=tmp[1]
    S[1][0]=tmp[1]
    S[1][1]=1
    
    u,s,vh=np.linalg.svd(S)
    
    s1=np.diag(s)
    
    D=np.sqrt(s1)
    K=np.dot(np.dot(u,D),u.transpose())
    
    H=np.zeros((3,3))
    H[0][0]=K[0][0]
    H[0][1]=K[0][1]
    H[1][0]=K[1][0]
    H[1][1]=K[1][1]
    H[2][2]=1
    
    
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
    
    print(xLength)
    print(yLength)
       
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

t1.append([123.0,82.0,1.0])
t1.append([516.0,74.0,1.0])
t1.append([124.0,307.0,1.0])
t1.append([489.0,382.0,1.0])

pts_image1=np.asarray(t1)

t2.append([1047.0,642.0,1.0])
t2.append([2582.0,80.0,1.0])
t2.append([927.0,2282.0,1.0])
t2.append([2423.0,2871.0,1.0])

pts_image2=np.asarray(t2)


H1=VLMatrix(pts_image1)
output=ProjectionImage(H1,image1)
cv2.imwrite('1_2stepP.jpg',output)

H2=VLMatrix(pts_image2)
output=ProjectionImage(H2,image2)
cv2.imwrite('2_2stepP.jpg',output)

tc=[]
tc.append([263.0,145.0,1.0])
tc.append([312.0,145.0,1.0])
tc.append([261.0,206.0,1.0])
tc.append([310.0,213.0,1.0])

pts=np.asarray(tc)

pts[0]=np.dot(H1,pts[0])
pts[0]=pts[0]/pts[0][2]
pts[1]=np.dot(H1,pts[1])
pts[1]=pts[1]/pts[1][2]
pts[2]=np.dot(H1,pts[2])
pts[2]=pts[2]/pts[2][2]
pts[3]=np.dot(H1,pts[3])
pts[3]=pts[3]/pts[3][2]

Hv=AffineMatrix(pts,pts_image1)
output=ProjectionImage(np.dot(np.linalg.pinv(Hv),H1),image1)
cv2.imwrite('1_2stepA.jpg',output)

td=[]
td.append([1194.0,920.0,1.0])
td.append([2168.0,699.0,1.0])
td.append([1123.0,1953.0,1.0])
td.append([2084.0,2159.0,1.0])

pt=np.asarray(td)

pt[0]=np.dot(H2,pt[0])
pt[0]=pt[0]/pt[0][2]
pt[1]=np.dot(H2,pt[1])
pt[1]=pt[1]/pt[1][2]
pt[2]=np.dot(H2,pt[2])
pt[2]=pt[2]/pt[2][2]
pt[3]=np.dot(H2,pt[3])
pt[3]=pt[3]/pt[3][2]

Hu=AffineMatrix(pt,pts_image2)
output=ProjectionImage(np.dot(np.linalg.pinv(Hu),H2),image2)
cv2.imwrite('2_2stepA.jpg',output)




