import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
from torchvision import transforms as T
from cv2 import cv2 
import random
import glob
import time
from math import fabs
torch.cuda.empty_cache() 

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier','stop sign'
]
#launch of two cameras
count=0
jl=0
jr=0
left = cv2.VideoCapture(1)
right = cv2.VideoCapture(0)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

left.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
left.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
right.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
right.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)


'''def cropHorizontal(image):
    return image[:,
            int((CAMERA_WIDTH-CROP_WIDTH)/2):
            int(CROP_WIDTH+(CAMERA_WIDTH-CROP_WIDTH)/2)]'''
if not left.isOpened() or not right.isOpened():
    print("Cannot open camera")
    exit()

net = cv2.dnn.readNetFromDarknet('C:/yolov3/yolov3.cfg', 'C:/yolov3/yolov3.weights')
classes = []
with open("C:/yolov3/coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
outputlayers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors= np.random.uniform(0,255,size=(len(classes),3))
#traffic sign detection
def pred(img,j,f):
    height,width,_ = img.shape
    blob = cv2.dnn.blobFromImage(img,0.00392,(128,128),(0,0,0),True,crop=False)
    net.setInput(blob)
    outs = net.forward(outputlayers)
    class_ids=[]
    confidences=[]
    boxes=[]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                center_x= int(detection[0]*width)
                center_y= int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                boxes.append([x,y,w,h]) #put all rectangle areas
                confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                class_ids.append(class_id) 
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            if label=='stop sign':
                j+=1
                if j%3==0:
                    f=True
            color = colors[i]
    return img,j,f

#photo collection
while True:
    fl=False
    fr=False
    files = os.listdir(path="C:/foto/calibration 2camera/stereoL")
    if not (left.grab() and right.grab()):
        print("No more frames")
        break
    # Capture frame-by-frame 00392
    '''retl, framel = left.read()
    retr, framer = right.read()'''
    _, framel = left.retrieve()
    _, framer = right.retrieve()
    framl,jl,fl=pred(framel,jl,fl)
    framr,jr,fr=pred(framer,jr,fr)
    if fl and fr:
        cv2.imwrite('C:/foto/calibration 2camera/stereoL/%d.png' % count,framl)
        cv2.imwrite('C:/foto/calibration 2camera/stereoR/%d.png' % count,framr)
    count +=1
    cv2.imshow('windowl', framl)
    cv2.imshow('window', framr)
    if cv2.waitKey(25) & 0xFF == ord('q')or len(files)>9:
        cv2.destroyAllWindows()
        break



    # if frame is read correctly ret is True
    '''if not retl or not retr:
        print("Can't receive frame (stream end?). Exiting ...")
        break'''


    # Display the resulting frame
    
print('END')    
# When everything done, release the capture

left.release()
right.release()
cv2.destroyAllWindows()
#work with saved images
def get_prediction(img_path, threshold=0.5):
  img = Image.open(img_path)
  transform = T.Compose([T.ToTensor()])
  img = transform(img)
  img = img.to(device)
  pred = model([img])
  flag1=False
  #pred_score = list(pred[0]['scores'].detach().numpy())
  pred_score = list(pred[0]['scores'])
  
  pred_t = [pred_score.index(x) for x in pred_score if x>threshold]
  global pred_boxes, k
  if pred_t!=[]:
      
    pred_t=pred_t[-1]
    
    
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'])]
    
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach())]
    pred_boxes = pred_boxes[:pred_t+1]

    k=1000  
    for j, c in enumerate(pred_class):
      if c == 'stop sign':
        k=j
        if k<len(pred_boxes):
          flag1=True
          break
  
  return pred_boxes,k,flag1

def narezka(img_path,box,l):
  picture = Image.open(img_path)
  width, height = picture.size
  #(797.6748, 504.3219), (1167.3717, 839.1529)
  lx=int(box[l][0][0])
  ly=int(box[l][0][1])
  rx=int(box[l][1][0])
  ry=int(box[l][1][1])

  for x in range(1,width,1):
    for y in range(1,height,1):
        if x<lx-15 or x>rx+15 or y<ly-15 or y>ry+15:
          picture.putpixel( (x,y), (0, 0, 0, 255))
  
  return cv2.cvtColor(np.asarray(picture), cv2.COLOR_RGB2BGR)

import itertools
def Punkt(img):
  orb = cv2.ORB_create(nfeatures=180)
  kp = orb.detect(img, None)
  kp, _ = orb.compute(img, kp)
  return kp
def sift(img):
  detector = cv2.SIFT_create(nfeatures=180)
  #descriptor = cv2.DescriptorExtractor_create("SIFT")
 
  skp = detector.detect(img, None)
  skp, _ = detector.compute(img, skp)
  #skp, sd = descriptor.compute(img, skp)
 
  #tkp = detector.detect(template)
  #tkp, td = descriptor.compute(template, tkp)
  return skp

def detectionRED(frame, spisok):
    global approx
    img=frame.copy()
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #cv2.imshow('frame_hsv',frame_hsv)
    mask1 = cv2.inRange(frame_hsv, (0,50,50), (30,225,225))
    mask2 = cv2.inRange(frame_hsv, (120,50,50), (255,225,225))
    frame_mask = cv2.bitwise_or(mask1, mask2 )
    #cv2.imshow('frame_mask',frame_mask)
    frame_dilate = cv2.dilate(frame_mask, None, iterations=2)
    #cv2.imshow('frame_dilate',frame_dilate)
    contours, _ = cv2.findContours(frame_dilate.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sort = spisok
    flag2=False
    for i in contours:
        area = cv2.contourArea(i)
        if 3000 < area < 100000000:  
            sort.append(i)
    for cnt in sort:
      epsilon = 0.01*cv2.arcLength(cnt, True)
      approx = cv2.approxPolyDP(cnt, epsilon, True) #Функция cv::approxPolyDP аппроксимирует кривую или многоугольник
      # другой кривой/многоугольником с меньшим количеством вершин так, чтобы расстояние между ними было меньше или равно заданной точности.
      
      if len(approx) == 8:
          flag2=True
          '''cv2.drawContours(img, [approx], 0, (0), 1)
          for i in range(len(approx)):
            img = cv2.circle(img, (approx[i][0][0],approx[i][0][1]), radius=0, color=(0, 0, 0), thickness=1)'''
          for i in range(len(approx)):
                if approx[i][0][0]<10 or approx[i][0][1]<10:
                  flag2=False
          break
    
    '''if len(approx)!=8:
        print(flag2)'''
    #cv2.imshow('img',img)
    #cv2.waitKey(0)
    return img, flag2, approx

def show_neuro_image(img_path, threshold=0.5, rect_th=1):
  boxes,q,flag1 = get_prediction(img_path, threshold)
  l=q  
  if flag1:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.rectangle(img, 
                      (int(boxes[l][0][0]),int(boxes[l][0][1])), 
                      (int(boxes[l][1][0]),int(boxes[l][1][1]))
                      ,color=(0, 255, 0), thickness=rect_th)
    plt.figure(figsize=(20,30))
    plt.imshow(img)
    plt.show()

def nearp(cernel,pts):
  d=[]
  r=[]
  flag=True
  for i in range(len(cernel)):
    d.append((cernel[i][0][0],cernel[i][0][1]))
  d=np.array(d)
  for i in range(len(d)):
    distances = np.linalg.norm(pts-d[i], axis=1)
    min_index = np.argmin(distances)
    r.append(list(pts[min_index]))
    if (fabs(d[i][0]-pts[min_index][0]) > 6 or fabs(d[i][1]-pts[min_index][1]) > 6):
      flag=False
  #print(cernel,'\n',r)
  return(r,flag)

def split(c):
  str=c.split('\\')
  return str[1]
def get_impoints(imagePaths,cern):
    razdelenieimg=[]
    t=[]
    for z, c in enumerate(imagePaths):
      if z>=0:
        b, index, flag1 = get_prediction(c)
        print(c)
        
        if flag1:
          s=[]          
          pix=narezka(c,b,index)
          n=split(c)
          path='C:/foto/calibration 2camera/stlap/'
          cv2.imwrite(path+n,pix)
          
          #kp = sift(pix)
          img, flag, cernel=detectionRED(pix,s)
          if flag:
            
            pts = harris(img)
            r,flag=nearp(cernel,pts)
            if flag==0:
              continue
            cern.append(r)   #вернуть для работы
            n=split(c)
            
            razdelenieimg.append(split(c))
            t.append(True)
            
            #pts = cv2.KeyPoint_convert(kp)
            
            '''for i in range(len(pts)):
              img = cv2.circle(img, (int(pts[i][0]),int(pts[i][1])), radius=0, color=(0, 0, 255), thickness=-1)'''
            for i in range(len(r)):
              img = cv2.circle(img, (int(r[i][0]),int(r[i][1])), radius=0, color=(0, 255, 0), thickness=-1)
             #img[int(pts[i][0]),int(pts[i][1])]=[0,255,0]
            
            #img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
            cv2.putText(img, "0", (cernel[0][0][0], cernel[0][0][1]), cv2.FONT_HERSHEY_COMPLEX, 1, 0,2)
            cv2.putText(img, "1", (cernel[1][0][0], cernel[1][0][1]), cv2.FONT_HERSHEY_COMPLEX, 1, 0,2)
                          
            PIL_image = Image.fromarray(np.uint8(img)).convert('RGB')
            plt.figure(figsize=(20,30))
            plt.imshow(PIL_image)
            plt.show()
            
      if z>57:
        break
    return cern, razdelenieimg, t
#409, 315

objp=np.array([[[0,0,0],[69,0,0],[118,49,0],[118,118,0],[69,167,0],[0,167,0],[-49,118,0],[-49,49,0]]])
objpp = np.zeros((1, 8, 3), np.float32)
for i in range(len(objp[0])):
  objpp[0][i][0] =objp[0][i][0]
  objpp[0][i][1] =objp[0][i][1]

objl=np.array([[[0,0,0],[69,0,0],[118,49,0],[118,118,0],[69,167,0],[0,167,0],[-49,118,0],[-49,49,0]]])
objll = np.zeros((1, 8, 3), np.float32)
for i in range(len(objp[0])):
  objll[0][i][0] =objl[0][i][0]
  objll[0][i][1] =objl[0][i][1]
Lobjpoints =[]
Robjpoints =[]

def find_centroids(dst, gray):
    ret, dst = cv2.threshold(dst, 0.0001 * dst.max(), 255, 0)
    dst = np.uint8(dst)



    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS , 400, 
                0.1)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(10,10), 
              (-1,-1),criteria)
    return corners
def harris(image):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray = np.float32(gray)
  dst = cv2.cornerHarris(gray, 16, 5, 0.04)
  dst = cv2.dilate(dst, None)
  #image[dst > 0.01*dst.max()] = [255, 0, 0]
  corners= find_centroids(dst,gray)
  for corner in corners:
    image[int(corner[1]), int(corner[0])] = [0, 0, 255]
  return corners
  '''cv2.imshow('dst', image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()'''

leftImageDir = 'C:/foto/calibration 2camera/stereoL'
imagePaths = glob.glob("{0}/*.png".format(leftImageDir))
p=[]
_, razdelenieimg,_ = get_impoints(imagePaths,p)
print(razdelenieimg)

leftImageDir = 'C:/foto/calibration 2camera/stereoR'
imagePaths = glob.glob("{0}/*.png".format(leftImageDir))
p=[]
f=[]
for i in razdelenieimg:
  f.append(leftImageDir+'\\'+i)
cernelR,razdelenieimg2,t=get_impoints(f,p)

for i in t:
  if i:
    Robjpoints.append(objpp)
  else:
    Robjpoints.append(objll)

cer = np.zeros((1, 8, 2), np.float32)

Rimgpoints=[]
for i in range(len(cernelR)):
  z=cer.copy()
  Rimgpoints.append(z)

for i in range(len(cernelR)):
  for j in range(8):
    Rimgpoints[i][0][j][0]=cernelR[i][j][0]
    Rimgpoints[i][0][j][1]=cernelR[i][j][1]

print(len(Rimgpoints))

img = cv2.imread(imagePaths[0])
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(Robjpoints, Rimgpoints, img.shape[::-1], None, None)
print(mtxR)

f=[]
for i in razdelenieimg2:
  f.append(leftImageDir+'\\'+i)
print(razdelenieimg2)
print(f)

leftImageDir = 'C:/foto/calibration 2camera/stereoL'
imagePaths = glob.glob("{0}/*.png".format(leftImageDir))
f=[]
p=[]
for i in razdelenieimg2:
  f.append(leftImageDir+'\\'+i)
cernelL, _,t2 = get_impoints(f,p)

for i in t2:
  if i:
    Lobjpoints.append(objpp)
  else:
    Lobjpoints.append(objll)

print(razdelenieimg)
print(razdelenieimg2)
print(len(razdelenieimg2))

cer = np.zeros((1, 8, 2), np.float32)
Limgpoints=[]
for i in range(len(cernelL)):
  a=cer.copy()
  Limgpoints.append(a)

for i in range(len(cernelL)):
  for j in range(8):
    Limgpoints[i][0][j][0]=cernelL[i][j][0]
    Limgpoints[i][0][j][1]=cernelL[i][j][1]

img = cv2.imread(imagePaths[0])
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(Lobjpoints, Limgpoints, img.shape[::-1], None, None)
print(mtxL)
print(distL)
print(mtxR)
print(distR)

from scipy.linalg import svd
def compute_H(objpoints, imgpoints):
  N = 7

  M = np.zeros((2*N, 9), dtype=np.float64)
  for i in range(N):
    X=objpoints[i][0]
    Y=objpoints[i][1]
    u=imgpoints[i][0]
    v=imgpoints[i][1]
    #print(u)
    row_1 = np.array([ -X, -Y, -1, 0, 0, 0, X*u, Y*u, u])
    row_2 = np.array([ 0, 0, 0, -X, -Y, -1, X*v, Y*v, v])
    M[2*i] = row_1
    M[(2*i) + 1] = row_2
  u, s, vh = np.linalg.svd(M)
  h = vh[np.argmin(s)]
  h = h.reshape(3, 3)
  #нормальное псевдорешение
  
  return h

H=[]
for i in range(len(cernelL)):
  #print(i)
  h=compute_H(Lobjpoints[i][0],cernelL[i])
  h.reshape(3, 3)
  H.append(h)
  #print(h,'\n')

def get_intrinsic_parameters(H_r):
    M = len(H_r)
    V = np.zeros((2*M, 6), np.float64)
    def v_pq(p, q, H):
        v = np.array([
                H[0, p]*H[0, q],
                H[0, p]*H[1, q] + H[1, p]*H[0, q],
                H[1, p]*H[1, q],
                H[2, p]*H[0, q] + H[0, p]*H[2, q],
                H[2, p]*H[1, q] + H[1, p]*H[2, q],
                H[2, p]*H[2, q]
            ])
        return v

    for i in range(M):
        H = H_r[i]
        V[2*i] = v_pq(p=0, q=1, H=H)
        V[2*i + 1] = np.subtract(v_pq(p=0, q=0, H=H), v_pq(p=1, q=1, H=H))
    # solve V.b = 0
    u, s, vh = np.linalg.svd(V)
    # print(u, "\n", s, "\n", vh)
    b = vh[np.argmin(s)]
    print("V.b = 0 Solution : ", b)

    # according to zhangs method
    vc = (b[1]*b[3] - b[0]*b[4])/(b[0]*b[2] - b[1]**2)
    l = b[5] - (b[3]**2 + vc*(b[1]*b[2] - b[0]*b[4]))/b[0]
    alpha = np.sqrt((l/b[0]))
    beta = np.sqrt(((l*b[0])/(b[0]*b[2] - b[1]**2)))
    gamma = -1*((b[1])*(alpha**2) *(beta/l))
    uc = (gamma*vc/beta) - (b[3]*(alpha**2)/l)

    '''print([vc,
            l,
            alpha,
            beta,
            gamma,
        uc])'''

    A = np.array([
            [alpha, gamma, uc],
            [0, beta, vc],
            [0, 0, 1.0],
        ])
    print("Intrinsic Camera Matrix is :")
    #print(A)
    return A
get_intrinsic_parameters(H)

print('Ro: ',len(Robjpoints),'\n','Lo: ',len(Lobjpoints),'\n','Ri: ',len(Rimgpoints),'\n','Li: ',len(Limgpoints),'\n')

print(distL)
print(distR)

distL=np.zeros((1,5))
distR=np.zeros((1,5))

print("Calibrating cameras together...")
(_, _, _, _, _, rotationMatrix, translationVector, _, _) = cv2.stereoCalibrate(
        Robjpoints, Limgpoints, Rimgpoints,
        mtxL, distL,
        mtxR, distR,
        img.shape[::-1], None, None, None, None,
        cv2.CALIB_USE_INTRINSIC_GUESS)


print("Rectifying cameras...")

(leftRectification, rightRectification, leftProjection, rightProjection,
        _, leftROI, rightROI) = cv2.stereoRectify(
                mtxL, distL,
                mtxR, distR,
                img.shape[::-1], rotationMatrix, translationVector,
                None, None, None, None, None,
                cv2.CALIB_ZERO_DISPARITY)

print("Saving calibration...")
outputFile = 'C:/foto/calibration 2camera/outputFileZnak'
leftMapX, leftMapY = cv2.initUndistortRectifyMap(
        mtxL, distL, leftRectification,
        leftProjection, img.shape[::-1], cv2.CV_32FC1)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(
        mtxR, distR, rightRectification,
        rightProjection, img.shape[::-1], cv2.CV_32FC1)

np.savez_compressed(outputFile, imageSize=img.shape[::-1],
        leftMapX=leftMapX, leftMapY=leftMapY, leftROI=leftROI,
        rightMapX=rightMapX, rightMapY=rightMapY, rightROI=rightROI)
