import numpy as np
import cv2
import math

def onestep(arrays1,arrays2):
    
    l1=np.cross(arrays2[0],arrays2[1])
    m1=np.cross(arrays2[1],arrays2[3])
    l1=l1/max(l1)
    m1=m1/max(m1)
    
    l2=np.cross(arrays2[1],arrays2[3])
    m2=np.cross(arrays2[3],arrays2[2])
    l2=l2/max(l2)
    m2=m2/max(m2)
    
    l3=np.cross(arrays2[3],arrays2[2])
    m3=np.cross(arrays2[2],arrays2[0])
    l3=l3/max(l3)
    m3=m3/max(m3)
    
    l4=np.cross(arrays2[2],arrays2[0])
    m4=np.cross(arrays2[0],arrays2[1])
    l4=l4/max(l4)
    m4=m4/max(m4)
    
    l5=np.cross(arrays1[0],arrays1[3])
    m5=np.cross(arrays1[1],arrays1[2])
    l5=l5/max(l5)
    m5=m5/max(m5)
    
    ta=[]
    tb=[]
    
    ta.append([l1[0]*m1[0],(l1[0]*m1[1]+l1[1]*m1[0])/2,l1[1]*m1[1],(l1[0]*m1[2]+l1[2]*m1[0])/2,(l1[1]*m1[2]+l1[2]*m1[1])/2])
    tb.append([-l1[2]*m1[2]])
    
    ta.append([l2[0]*m2[0],(l2[0]*m2[1]+l2[1]*m2[0])/2,l2[1]*m2[1],(l2[0]*m2[2]+l2[2]*m2[0])/2,(l2[1]*m2[2]+l2[2]*m2[1])/2])
    tb.append([-l2[2]*m2[2]])
    
    ta.append([l3[0]*m3[0],(l3[0]*m3[1]+l3[1]*m3[0])/2,l3[1]*m3[1],(l3[0]*m3[2]+l3[2]*m3[0])/2,(l3[1]*m3[2]+l3[2]*m3[1])/2])
    tb.append([-l3[2]*m3[2]])
    
    ta.append([l4[0]*m4[0],(l4[0]*m4[1]+l4[1]*m4[0])/2,l4[1]*m4[1],(l4[0]*m4[2]+l4[2]*m4[0])/2,(l4[1]*m4[2]+l4[2]*m4[1])/2])
    tb.append([-l4[2]*m4[2]])
    
    ta.append([l5[0]*m5[0],(l5[0]*m5[1]+l5[1]*m5[0])/2,l5[1]*m5[1],(l5[0]*m5[2]+l5[2]*m5[0])/2,(l5[1]*m5[2]+l5[2]*m5[1])/2])
    tb.append([-l5[2]*m5[2]])
    
    A=np.asarray(ta)
    b=np.asarray(tb)
    
    tmp=np.dot(np.linalg.pinv(A),b)
    tmp=tmp/np.max(tmp)
    
    S=np.zeros((2,2))
    S[0][0]=tmp[0]
    S[0][1]=tmp[1]/2
    S[1][0]=tmp[1]/2
    S[1][1]=tmp[2]
    
    u,s,vh=np.linalg.svd(S)
    
    s1=np.diag(s)
    
    D=np.sqrt(s1)
    K=np.dot(np.dot(u,D),u.transpose())
    
    tmp1=np.array([tmp[3]/2,tmp[4]/2])
    
    print(K)
    print(tmp1)
    
    v=np.dot(np.linalg.pinv(K),tmp1)
    
    H=np.zeros((3,3))
    H[2][2]=1
    H[0][0]=K[0][0]
    H[0][1]=K[0][1]
    H[1][0]=K[1][0]
    H[1][1]=K[1][1]
    H[2][0]=v[0]
    H[2][1]=v[1]
    
    
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
    
    print(WorldA)
    print(WorldB)
    print(WorldC)
    print(WorldD)
    
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

tc=[]
tc.append([263.0,145.0,1.0])
tc.append([312.0,145.0,1.0])
tc.append([261.0,206.0,1.0])
tc.append([310.0,213.0,1.0])
pts=np.asarray(tc)

td=[]
td.append([1194.0,920.0,1.0])
td.append([2168.0,699.0,1.0])
td.append([1123.0,1953.0,1.0])
td.append([2084.0,2159.0,1.0])

pt=np.asarray(td)


H1=onestep(pts,pts_image1)
output=ProjectionImage(np.linalg.pinv(H1),image1)
cv2.imwrite('1_1step.jpg',output)

H2=onestep(pt,pts_image2)
output=ProjectionImage(np.linalg.pinv(H2),image2)
cv2.imwrite('2_1step.jpg',output)









