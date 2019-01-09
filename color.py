import cv2
import face_recognition
import numpy as np
def rounde(p):
    if p>255:
        p=255
    if p<0:
        p=0
    return p
def color(b,g,r,img):
    img2=np.zeros((len(img),len(img[0]),3),dtype=np.uint8)
    for i in range(len(img)):
        for j in range(len(img[0])):
            img2[i][j][0]=int(img[i][j][0])
            img2[i][j][1]=int(img[i][j][1])
            img2[i][j][2]=int(img[i][j][2])
    img3=np.zeros((len(img),len(img[0]),3),dtype=np.uint8)
    for i in range(len(img)):
        for j in range(len(img[0])):
            img3[i][j][0]=img[i][j][0]
            img3[i][j][1]=img[i][j][1]
            img3[i][j][2]=img[i][j][2]
    face_list=face_recognition.face_landmarks(img)
    p=[]
    for k in face_list[0]["top_lip"]:
        p.append([k[0],k[1]])
    pts=np.array(p,np.int32)
    pts=pts.reshape((-1,1,2))
    cv2.fillPoly(img,[pts],(255,255,0))
    p=[]
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j][0]==255 and img[i][j][1]==255 and img[i][j][2]==0:
                p.append((i,j))
    d=0
    u1=-0.147*r-0.289*g+0.436*b
    v1=0.615*r-0.515*g-0.1*b
    for k in p:
        i=k[0]
        j=k[1]
        y=0.299*int(img2[i][j][2])+0.114*int(img2[i][j][0])+0.587*int(img2[i][j][1])
        y2=y
        u2=u1
        v2=v1
        r_new=y2+1.14*v2
        g_new=y2-0.39*u2-0.58*v2
        b_new=y2+2.03*u2
        r_new=rounde(r_new)
        g_new=rounde(g_new)
        b_new=rounde(b_new)
        img2[i][j][0]=b_new
        img2[i][j][1]=g_new
        img2[i][j][2]=r_new
    p=[]
    for k in face_list[0]["bottom_lip"]:
        p.append([k[0],k[1]])
    pts=np.array(p,np.int32)
    pts=pts.reshape((-1,1,2))
    cv2.fillPoly(img3,[pts],(255,255,0))
    p=[]
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img3[i][j][0]==255 and img3[i][j][1]==255 and img3[i][j][2]==0:
                p.append((i,j))
    u1=-0.147*r-0.289*g+0.436*b
    v1=0.615*r-0.515*g-0.1*b
    for k in p:
        i=k[0]
        j=k[1]
        y=0.299*int(img2[i][j][2])+0.114*int(img2[i][j][0])+0.587*int(img2[i][j][1])
        y2=y
        u2=u1
        v2=v1
        r_new=y2+1.14*v2
        g_new=y2-0.39*u2-0.58*v2
        b_new=y2+2.03*u2
        r_new=rounde(r_new)
        g_new=rounde(g_new)
        b_new=rounde(b_new)
        img2[i][j][0]=b_new
        img2[i][j][1]=g_new
        img2[i][j][2]=r_new
    cv2.imwrite("10_t.png",img2)
b=91
g=111
r=223
img=cv2.imread("tar.jpg")
color(91,111,223,img)
