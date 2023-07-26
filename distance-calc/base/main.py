import cv2
from ultralytics import YOLO
from matplotlib import pyplot as plt

s=0.0038
p=0.0074
f=850

def focalLength(w,d,W):
    return w*d/W

def distance(f,w,W):
    return W*f/w



objects=["person","bicyle","car","motorcycle","bus","truck","traffic light","stop sign","bench","chair","couch","bottle"]
# change objects to ints so that checking is even faster

def calculate(img_src):
    # Image
    image=cv2.imread(img_src)

    # Model
    model=YOLO("../model/yolov8l.pt")

    # Prediction
    yolo_pred=model.predict(img_src)
    tensor=yolo_pred[0].boxes.data
    names=yolo_pred[0].names
    # print(names)

    # Results and Calculation
    for element in tensor:
        label=names[int(element[5])]
        if(label in objects and element[4]>0.5):
            x1,y1,x2,y2=int(element[0]),int(element[1]),int(element[2]),int(element[3])

            start_corner=(x1,y1)
            end_corner=(x2,y2)
            dis=distance(f,y2-y1,1.5)
            
            cv2.rectangle(image,start_corner,end_corner,(0,255,0),2)

            text_size=cv2.getTextSize(label,cv2.FONT_HERSHEY_COMPLEX,0.5,1)[0]
            text_x=x1
            text_y=y1-5
            cv2.rectangle(image, (text_x, text_y), (text_x + text_size[0], text_y - text_size[1]), (0,255,0), -1)
            if(label=="person" or label=="motorcycle" or label=="car"):
                st=dis*100
                st=str(st)+"cm"
                cv2.putText(image,st , (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                # print(dis*100,"cm")
            else:
                cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

# cv2.imshow("Predicted image",image)
plt.figure(1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()





