import imutils
import json
import time
import cv2
import datetime

#raspi specific libraries
from picamera.array import PiRGBArray
from picamera import PiCamera
import RPi.GPIO as GPIO

#GPIO specific setup
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(7, GPIO.OUT)
GPIO.setup(11, GPIO.OUT)
GPIO.output([7,11],0)
#PiCamera specific setup
with open('camera_conf.json') as json_file:
    camconf = json.load(json_file)

picam = PiCamera()
picam.resolution = tuple(camconf["resolution"])
picam.framerate = camconf["fps"]
rawStream = PiRGBArray(picam, size=tuple(camconf["resolution"]))
print("Camera is getting configured")
time.sleep(camconf["camera_warmup_time"])

avgBackgroundFrame = None
peopleCounter = 0

for fr in picam.capture_continuous(rawStream, format="bgr", use_video_port=True):
    frame = fr.array
    timestamp = datetime.datetime.now()
    text = "No one in sight"
    frame = imutils.resize(frame, width=500)
    grayScaledFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    noiseRemovedFrame = cv2.GaussianBlur(grayScaledFrame, (21, 21), 0)

    if avgBackgroundFrame is None:
        print("setting up background frame")
        avgBackgroundFrame = noiseRemovedFrame.copy().astype("float")
        rawStream.truncate(0)
        continue
    
    cv2.accumulateWeighted(noiseRemovedFrame, avgBackgroundFrame, 0.5)
    frameDelta = cv2.absdiff(noiseRemovedFrame, cv2.convertScaleAbs(avgBackgroundFrame))

    thresh = cv2.threshold(frameDelta, camconf["delta_thresh"], 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for con in contours:
        if cv2.contourArea(con) < camconf["min_area"]:
            continue

        (x, y, w, h) = cv2.boundingRect(con)
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)
        text = "Motion detected in hallway"

    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(frame, "Status: {}".format(text), (10, 20),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.35, (0, 0, 255), 1)    
    
    if text == "Motion detected in hallway":
        peopleCounter += 1
        GPIO.output([7,11],1)
    else:
        GPIO.output([7,11],0)
    
    if camconf["show_video"]:
        cv2.imshow("Hallway Cam", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            GPIO.cleanup()
            break
    rawStream.truncate(0)
    print(peopleCounter)
GPIO.cleanup()
