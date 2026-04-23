import cv2
import requests
import threading
import time

CAMERA_URL = "http://192.168.29.83:4747/video"
SERVER_URL = "http://192.168.29.151:5000/alert"

paused = False

# 🔥 BACKGROUND THREAD (checks status)
def check_status():
    global paused
    while True:
        try:
            res = requests.get(SERVER_URL, timeout=1)
            status = res.json()["status"]

            if status == "QUARANTINED":
                paused = True
            else:
                paused = False

        except:
            pass

        time.sleep(2)   # check every 2 sec only


# 🔥 Start background thread
threading.Thread(target=check_status, daemon=True).start()


# 🔥 Camera setup
cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    print("❌ Camera connection failed")
    exit()

print("📷 Camera running smoothly")

last_frame = None

while True:

    if not paused:
        ret, frame = cap.read()

        if ret:
            last_frame = frame
        else:
            print("⚠ Frame lost")
            continue

    # show last frame (freeze effect)
    if last_frame is not None:
        cv2.imshow("AEIS Camera Feed", last_frame)

    # no sleep → smooth FPS
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()