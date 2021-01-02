import cv2

webcam = cv2.VideoCapture(0)
#webcam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
#webcam.set(28,200)

c=0

while(True):
    ret, frame = webcam.read()
    frame=cv2.resize(frame, dsize=(640, 640), interpolation=cv2.INTER_LINEAR)
    #cv2.circle(frame, (320,240), 2, (0, 0, 255), 1)
    cv2.imshow("img",frame)

    if cv2.waitKey(1) == ord('c'):
        
        cv2.imwrite("../hand/h_img_%d.jpg" % c, img=frame)
        print("chal kak", c)
        c+=1