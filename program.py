import cv2
import sys
import os
import math
import shutil

class Face(object):
    area = 0
    path = ''
    def __init__(self, width, height, path):
        self.area = width * height
        self.path = path

    def scaleFactor(self, desiredArea):
        return math.sqrt(1.0 * desiredArea / self.area)

cascPath = 'haarcascade_frontalface_default.xml'

faceCascade = cv2.CascadeClassifier(cascPath)
facesList = []
directory = '/home/kyle/Projects/Photo A Day/Photos/'
subdirectoryName = 'temp/'
subdirectory = directory + subdirectoryName
images = os.listdir(directory)

try:
    os.makedirs(subdirectory)
except OSError:
    if not os.path.isdir(subdirectory):
        raise
skipped = 0
for imageName in images:
    # Read the image
    imagePath = directory + imageName
    #print imagePath
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(400, 400),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    newPath =  subdirectory + imageName
    largestFace = None

    for (x, y, w, h) in faces:
        if (largestFace == None or largestFace.area < w * h):
            largestFace = Face(w, h, newPath)
            largestFace.x = x
            largestFace.y = y
            largestFace.w = w
            largestFace.h = h
    if (largestFace != None):
        facesList.append(largestFace)
        # copy files to new temp directory to not destroy originals
        shutil.copyfile(imagePath, newPath)
    else:
        skipped += 1
print "Could not find faces in {0} photos".format(skipped)

for face in facesList:
    scaleFactor = face.scaleFactor(750 * 750)
    image = cv2.imread(face.path)
    (height, width) = image.shape[:2]
    newDim = (int(width * scaleFactor), int(height * scaleFactor))

    resized = cv2.resize(image, newDim, interpolation = cv2.INTER_AREA)

    resizedFaceY = int(face.y * scaleFactor)
    resizedFaceX = int(face.x * scaleFactor)
    resizedFaceH = int(face.h * scaleFactor)
    resizedFaceW = int(face.w * scaleFactor)
    padding = 100
    topBorder = resizedFaceY - (1080 / 2) + resizedFaceH / 2
    bottomBorder = topBorder + (1080)
    leftBorder = resizedFaceX - (1920 / 2) + resizedFaceW / 2
    rightBorder = leftBorder + (1920)
    print "Name: {0} \t Scale Factor: {1}".format(face.path[-10:], scaleFactor)
    if (leftBorder < 0 or bottomBorder < 0):
        print "skipping {0}".format(face.path)
        os.remove(face.path)
    else:
        cropped = resized[topBorder:bottomBorder, leftBorder:rightBorder]
        cv2.imwrite(face.path, cropped)

#def renameAsNumbers(directory):
    # TODO rename files in given directory 001, 002, etc.

#cleanup
#shutil.rmtree(subdirectory)
