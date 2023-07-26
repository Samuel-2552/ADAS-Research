import cv2
from ultralytics import YOLO
from matplotlib import pyplot as plt
# from calibration import focal_length

focal_length=1400
# print(focal_length)

# change objects to ints so that checking is even faster
objects= ["person","bicyle","car","motorcycle","bus","truck","bird","cat","dog","horse","sheep","cow"]


objects_height = {
    "person": 1.5,
    "bicyle": 0.6,
    "car": 1.5,
    "motorcycle": 0.65,
    "bus": 4,
    "truck": 4,
    "bird": 0.12,
    "cat": 0.25,
    "dog": 0.45,
    "horse": 1.5,
    "sheep": 0.85,
    "cow": 1.4
}

def distanceMts(focal_length, image_height_of_object_pxl, actual_height_of_object_meters):
    return (actual_height_of_object_meters*focal_length)/image_height_of_object_pxl


def calculate(image):
    # Model
    model=YOLO("../yolov8l.pt")

    # Prediction
    yolo_pred=model.predict(image)
    tensor=yolo_pred[0].boxes.data
    names=yolo_pred[0].names

    # Results and Calculation
    for element in tensor:
        label=names[int(element[5])]
        if(label in objects and element[4]>0.5):
            x1,y1,x2,y2=int(element[0]),int(element[1]),int(element[2]),int(element[3])

            start_corner=(x1,y1)
            end_corner=(x2,y2)
            print(y2-y1)
            dis=round(distanceMts(focal_length,y2-y1,objects_height[label]),2)

            cv2.rectangle(image,start_corner,end_corner,(0,255,0),2)
            text_size=cv2.getTextSize(str(dis),cv2.FONT_HERSHEY_COMPLEX,0.5,1)[0]
            text_x=x1
            text_y=y1-5
            cv2.rectangle(image, (text_x, text_y), (text_x + text_size[0], text_y - text_size[1]), (0,255,0), -1)
            st=str(dis)+"m"
            cv2.putText(image,st , (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return image

# Note- The sample input images from 12 to 21 are not calibrated for the primary camera as those images are randomly taken just for testing

# Image Processing

img_range_st=4
img_range_end=5

for i in range(img_range_st,img_range_end):
    img_src="../input/images/"+str(i)+".jpg"
    img_dst="../output/images/"+str(i)+".jpg"

    image=cv2.imread(img_src)

    image=calculate(image)
    cv2.imwrite(img_dst,image)

# plt.figure(1)
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()


# Video Processing

# vid_src="../input/videos/vidtest.mp4" # 0 for default camera
# video_capture=cv2.VideoCapture(vid_src)
# frame_width = int(video_capture.get(3))
# frame_height = int(video_capture.get(4))
# fps = video_capture.get(cv2.CAP_PROP_FPS)

# output_path = '../output/videos/out_vidtest.avi'
# output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# while True:
#     ret,frame=video_capture.read()
#     if not ret:
#         break
#     output=calculate(frame)
#     output_video.write(output)

# video_capture.release()
# output_video.release()


# End
cv2.waitKey(0)
cv2.destroyAllWindows()





