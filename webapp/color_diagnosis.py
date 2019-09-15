import cv2
import numpy as np

def cascade_rectangle(img):
    path='static/cascade/'
    cascades = ['haarcascade_fullbody.xml','haarcascade_lowerbody.xml']
    input=[]
    for cascade in cascades:
        load_cascade = cv2.CascadeClassifier(path+cascade)
        coordinate = load_cascade.detectMultiScale(img)
        #print(cascade,coordinate)
        for i in range(len(coordinate)):
            if len(coordinate[i])>0:
                x,y,w,h = coordinate[i]
                input.append(output_color(img,x,y,w,h))
                if cascade=='haarcascade_fullbody.xml':
                    h=h//2                
                cv2.rectangle(img,(x,y+30),(x+w,y+h-30),(255,0,0),2)

    input2 = [flatten for inner in input for flatten in inner]
    print(input2)
    return img,input2

def output_color(img,x,y,w,h):
    imgCrop = img[x:x+w, y+30:y+h-30,:]
    Median=[]
    for i in range(3):
        Median.append(np.median(imgCrop[:,:,i]))
            
    Median[2],Median[0] = Median[0],Median[2]
    return Median

