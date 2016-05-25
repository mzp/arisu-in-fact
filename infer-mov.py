from learn import infer
import shutil
import sys
import cv2
import os.path

cascade_file = "./lbpcascade_animeface.xml"
cascade = cv2.CascadeClassifier(cascade_file)

def detect(image):
    global cascade
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    for (x, y, w, h) in faces:
        yield(x, y, w, h)

input = cv2.VideoCapture(sys.argv[1])

width = input.get(cv2.CAP_PROP_FRAME_WIDTH)
height = input.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = input.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter('output.avi',
    fourcc,
    int(fps),
    (int(width), int(height)))

i = 0
while True:
    ret, frame = input.read()

    i += 1
    if ret == True:
        for (x,y, w,h) in detect(frame):
            crop = frame[y:y+h, x:x+w]
            pred = infer('model', 2, crop)
            if pred == 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        output.write(frame)
        cv2.imwrite('tmp/mov%d.png' % i, frame)
        print('mov%d' % i)
    else:
        break
