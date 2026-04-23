import cv2
import time

camera_url = "http://192.168.29.83:4747/video"

while True:

    cap = cv2.VideoCapture(camera_url)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Camera disconnected... reconnecting")
            cap.release()
            break   # break inner loop → reconnect

        cv2.imshow("AEIS Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    time.sleep(0.2)  # wait before reconnect