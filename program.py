import cv2
import sys
import os
import math
import shutil
import exifread
import datetime
from ffmpy import FFmpeg

class FaceImage(object):
    cascPath = 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)
    area = 0
    path = ''
    faceFound = False
    def __init__(self, path):
        self.update(path)

    def update(self, path):
        self.path = path
        # Read the image
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(400, 400),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        largestArea = None
        largestIndex = None
        index = 0
        for (x, y, w, h) in faces:
            if (largestArea == None or largestArea < w * h):
                largestIndex = index
                largestArea = w * h
            index += 1

        if (largestIndex != None):
            self.faceFound = True
            self.faceX = x
            self.faceY = y
            self.faceW = w
            self.faceH = h

    def move(self, newPath):
        # copy files to new temp directory to not destroy originals
        shutil.copyfile(self.path, newPath)
        self.path = newPath

    def calcScaleFactor(self, desiredArea):
        return math.sqrt(1.0 * desiredArea / self.getArea())

    def getArea(self):
        return self.faceH * self.faceW

    def getDateTaken(self):
        f = open(self.path, 'rb')
        tags = exifread.process_file(f)
        # TODO add time into the method instead of just chopping it off
        date = datetime.datetime.strptime(str(tags["EXIF DateTimeOriginal"])[:-9], '%Y:%m:%d')
        return date.strftime("%B %d")


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

def readFaces():
    skipped = 0
    for index, imageName in enumerate(images, start = 1):
        imagePath = directory + imageName
        newPath = subdirectory + imageName

        faceImage = FaceImage(imagePath)

        if (faceImage.faceFound):
            faceImage.move(newPath)
            facesList.append(faceImage)
        else:
            skipped += 1
        if (index % 10 == 0):
            print '{0}/{1} files processed. {2} skipped.'.format(index, len(images), skipped)

    print "Could not find faces in {0} photos".format(skipped)

def writeText(image, text):
    origin = (1000, 1000)
    color = (255, 255, 255)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 5, color, 4, cv2.CV_AA)
    return image

def resizeAndCrop():
    count = 1
    skipped = 0
    for index, faceImage in enumerate(facesList, start = 1):
        scaleFactor = faceImage.calcScaleFactor(1000 * 1000)
        image = cv2.imread(faceImage.path)
        (height, width) = image.shape[:2]
        newDim = (int(width * scaleFactor), int(height * scaleFactor))

        resized = cv2.resize(image, newDim, interpolation = cv2.INTER_AREA)

        resizedFaceY = int(faceImage.faceY * scaleFactor)
        resizedFaceX = int(faceImage.faceX * scaleFactor)
        resizedFaceH = int(faceImage.faceH * scaleFactor)
        resizedFaceW = int(faceImage.faceW * scaleFactor)

        desiredHeight = 1080
        desiredWidth = 1920

        topBorder = resizedFaceY - (desiredHeight / 2) + resizedFaceH / 2
        bottomBorder = topBorder + (desiredHeight)
        leftBorder = resizedFaceX - (desiredWidth / 2) + resizedFaceW / 2
        rightBorder = leftBorder + (desiredWidth)

        cropped = resized[topBorder:bottomBorder, leftBorder:rightBorder]
        (height, width) = cropped.shape[:2]

        if (height != desiredHeight or width != desiredWidth):
            os.remove(faceImage.path)
            skipped += 1
        else:
            image = writeText(cropped, faceImage.getDateTaken())
            cv2.imwrite(faceImage.path, image)
        if (index % 10 == 0):
            print '{0}/{1} images processed. {2} skipped'.format(index, len(facesList), skipped)

# necessary for outputting to video
def renameAsNumbers(directory):
    images = os.listdir(directory)
    images.sort()
    numImages = len(images)
    numImagesStrLen = len("{0}".format(numImages))

    for index, image in enumerate(images, start = 1):
        filetype = image[-4:]
        newName = "{0}".format(index)
        while (len(newName) < numImagesStrLen):
            newName = "0" + newName
        os.rename(directory + image, "{0}{1}{2}".format(directory, newName, filetype))

def createVideo(directory):
    os.chdir(directory)
    images = os.listdir(directory)
    length = len(os.path.splitext(images[0])[0])
    ff = FFmpeg(
        inputs={},
        outputs={'../out.mp4': '-framerate 3 -i %0{0}d.jpg -c:v libx264 -r 30 -pix_fmt yuv420p'.format(length)}
    )
    ff.run()
    os.chdir('..')
    filepath = os.path.abspath(os.curdir) + '/out.mp4'
    print 'Your video is complete and located at {0}'.format(filepath)

readFaces()
resizeAndCrop()
print 'Check photos in {0} and delete or manually fix any that did not resize properly.'.format(subdirectory)
raw_input("Press Enter to continue...")
renameAsNumbers(subdirectory)
createVideo(subdirectory)

#cleanup
shutil.rmtree(subdirectory)
