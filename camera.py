#!/usr/bin/env python3
import numpy as np
import cv2
from typing import NamedTuple

class Point(NamedTuple):
    x: int
    y: int
    z: int


cap = cv2.VideoCapture(0)
cascPath = "haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPath)


def drawline(img, start: Point, end: Point, observer: Point):
    if start.z<observer.z and end.z<observer.z:
        zscalestart = start.z/(observer.z - start.z)
        zscaleend = end.z/(observer.z - end.z)
        x1 = int(start.x + (start.x - observer.x) * zscalestart)
        y1 = int(start.y + (start.y - observer.y) * zscalestart)
        x2 = int(end.x + (end.x - observer.x) * zscaleend)
        y2 = int(end.y + (end.y - observer.y) * zscaleend)
        print(zscalestart, zscaleend, x1, y1, x2, y2)
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

def point2screen(pt: Point, obs: Point): 
    if pt.z<obs.z:
        zscale = pt.z/(obs.z - pt.z)
        x = int(pt.x + (pt.x - obs.x) * zscale)
        y = int(pt.y + (pt.y - obs.y) * zscale)
        return (x,y)
    raise ValueError


def face2observer(x,y,w,h) -> Point:
    xc=int(700-(x+w/2.0))
    yc=int(y+h/3.0)
    zc=int(120000/h)
    return Point(x=xc, y=yc, z=zc)

frame0 = np.zeros((1000,1200,3), np.uint8)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
#        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    #print(f"Found: {faces}")
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 255, 255), 2)
        obs = face2observer(x,y,w,h)
        try:
            xp,yp = point2screen(Point(300,300,300), obs)
            xp1,yp1 = point2screen(Point(350,350,300), obs)
            print(xp,yp,xp1,yp1)
            face = frame[x:x+w,y:y+h,:]
            resized = cv2.resize(face, (xp1-xp, yp1-yp), interpolation = cv2.INTER_AREA)
            frame0[xp:xp1,yp:yp1,:] = resized
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        except Exception as e:
            print(e)
#        drawline(gray, Point(300, 300, 0), Point(300,300,300), obs)
    cv2.imshow('frame', frame)
#    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

