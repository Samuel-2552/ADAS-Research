import cv2
from ultralytics import YOLO
from matplotlib import pyplot as plt

f=850

def focalLength(w,d,W):
    return w*d/W

def distance(f,w,W):
    return W*f/w

def detect(image):
    model=YOLO("../model/yolov8n.pt")
    yolo_pred=model.predict(image)
    objects=["person","bicyle","car","motorcycle","bus","truck","traffic light","stop sign","bench","chair","couch","bottle"]
    # change objects to ints so that checking is even faster

    tensor=yolo_pred[0].boxes.data
    names=yolo_pred[0].names

    for element in tensor:
        label=names[int(element[5])]
        if(label in objects and element[4]>0.5):
            # x,y,width,height=element.xyxy
            x1,y1,x2,y2=int(element[0]),int(element[1]),int(element[2]),int(element[3])
            # if(label=="bottle"):
                # pic_width=x2-x1
                # act_dis=2.4384
                # act_width=0.38
                # f=focalLength(pic_width,s,act_width)
                # f=focalLength(pic_width,act_dis,act_width)
                # print(f)
            # print(pic_width)
            # print(label," ",x1," ",y1," ",x2," ",y2)
            start_corner=(x1,y1)
            end_corner=(x2,y2)
            cv2.rectangle(image,start_corner,end_corner,(0,255,0),2)
            dis=distance(f,x2-x1,0.5)
            text_size=cv2.getTextSize(label,cv2.FONT_HERSHEY_COMPLEX,0.5,1)[0]
            text_x=x1
            text_y=y1-5
            cv2.rectangle(image, (text_x, text_y), (text_x + text_size[0], text_y - text_size[1]), (0,255,0), -1)
            if(label=="person" or label=="handbag" or label=="backpack" or label=="laptop" or label=="tv"):
                st=dis*100
                st=str(st)+"cm"
                cv2.putText(image,st , (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            # print(dis*100,"cm")
            else:
                cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return image

video_capture=cv2.VideoCapture("../images/vidtest2.mp4")
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
fps = video_capture.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
output_path = '../images/output_video2.avi'
output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

while True:
    ret,frame=video_capture.read()
    if not ret:
        break
    output=detect(frame)
    output_video.write(output)

video_capture.release()
output_video.release()
cv2.destroyAllWindows()





