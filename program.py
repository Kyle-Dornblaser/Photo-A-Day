import cv2
import sys
import os
import math
import shutil
import exifread
import datetime
from ffmpy import FFmpeg
import numpy as np

class FaceImage(object):
    cascPath = 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)

    def __init__(self, path):
        self.faceFound = False
        self.update(path)

    def update(self, path):
        self.path = path
        # Read the image
        image = cv2.imread(path)
        (self.height, self.width) = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        smallFace = (100, 100)
        mediumFace = (250, 250)
        largeFace = (400, 400)
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = largeFace,
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        if (len(faces) == 0):
            print 'Checking for medium faces.'
            faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor = 1.1,
                minNeighbors = 5,
                minSize = mediumFace,
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
        return math.sqrt(1.0 * desiredArea / self.getFaceArea())

    def getFaceArea(self):
        return self.faceH * self.faceW

    def getDateTaken(self):
        f = open(self.path, 'rb')
        tags = exifread.process_file(f)
        # TODO add time into the method instead of just chopping it off
        date = datetime.datetime.strptime(str(tags["EXIF DateTimeOriginal"])[:-9], '%Y:%m:%d')
        return date.strftime("%B %d")

def createTempDir(dirName):
    try:
        os.makedirs(subdirectory)
        print 'Creating temporary directory.'
    except OSError:
        if not os.path.isdir(subdirectory):
            raise

def readFaces(images, directory, subdirectory):
    faceImagesList = []
    skipped = 0
    for index, imageName in enumerate(images, start = 1):
        imagePath = directory + imageName
        newPath = subdirectory + imageName

        try:
            faceImage = FaceImage(imagePath)

            if (faceImage.faceFound):
                faceImage.move(newPath)
                faceImagesList.append(faceImage)
            else:
                skipped += 1
        except AttributeError:
            print '{0} is not an image. Skipping.'.format(imageName)
            skipped += 1
        if (index % 10 == 0):
            print '{0}/{1} files processed. {2} skipped.'.format(index, len(images), skipped)

    print "Could not find faces in {0} photos".format(skipped)
    return faceImagesList

def writeText(image, text):
    bottomRight = (1920 - 100, 1080 - 100)
    textLength = len(text)
    charWidth = 102
    # bad right alignment logic
    bottomRight = (1920 - 100 - textLength * charWidth, 1080 - 100)
    origin = bottomRight
    color = (255, 255, 255)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 5, color, 4, cv2.CV_AA)
    return image

def largestFaceArea():
    largest = None
    faceAreas = []
    for index, faceImage in enumerate(faceImagesList):
        faceAreas.append(faceImage.getFaceArea())
        if (largest == None or largest.getFaceArea() < faceImage.getFaceArea()):
            largest = faceImage

    faceAreasNP = np.array(faceAreas)
    np.sort(faceAreasNP)
    standardDev = np.std(faceAreasNP)
    mean = np.mean(faceAreasNP)

    # return two standard devations if largest is larger
    if (largest.getFaceArea() > mean + (2 * standardDev)):
        return mean + (2 * standardDev)
    else:
        return largest.getFaceArea()

def resizeAndCrop(faceImagesList):

    largest = largestFaceArea()

    desiredHeight = 1080
    desiredWidth = 1920

    wiggleRoom = 0.0
    heightScaleFactor = 1.0 * desiredHeight / faceImagesList[0].height
    widthScaleFactor = 1.0 * desiredWidth / faceImagesList[0].width

    #print 2250 * heightScaleFactor
    #print 3000 * widthScaleFactor
    #print largestFaceH * heightScaleFactor
    #print largestFaceH * widthScaleFactor

    if (heightScaleFactor >= widthScaleFactor):
        largestArea = (heightScaleFactor ** 2 + wiggleRoom) * largest
    else:
        largestArea = (widthScaleFactor ** 2 + wiggleRoom) * largest

    desiredFaceArea = largestArea

    skipped = 0
    for index, faceImage in enumerate(faceImagesList, start = 1):

        scaleFactor = faceImage.calcScaleFactor(desiredFaceArea)
        image = cv2.imread(faceImage.path)
        (height, width) = image.shape[:2]
        newDim = (int(width * scaleFactor), int(height * scaleFactor))

        #print newDim

        resized = cv2.resize(image, newDim, interpolation = cv2.INTER_AREA)

        resizedFaceY = int(faceImage.faceY * scaleFactor)
        resizedFaceX = int(faceImage.faceX * scaleFactor)
        resizedFaceH = int(faceImage.faceH * scaleFactor)
        resizedFaceW = int(faceImage.faceW * scaleFactor)

        topBorder = resizedFaceY - (desiredHeight / 2) + resizedFaceH / 2
        bottomBorder = topBorder + (desiredHeight)
        leftBorder = resizedFaceX - (desiredWidth / 2) + resizedFaceW / 2
        rightBorder = leftBorder + (desiredWidth)

        cropped = resized[topBorder:bottomBorder, leftBorder:rightBorder]
        (height, width) = cropped.shape[:2]

        if (height != desiredHeight or width != desiredWidth):
            #print 'H: {0} \t W: {1}'.format(height, width)
            os.remove(faceImage.path)
            skipped += 1
        else:
            image = writeText(cropped, faceImage.getDateTaken())
            cv2.imwrite(faceImage.path, image)
        if (index % 10 == 0):
            print '{0}/{1} images processed. {2} skipped'.format(index, len(faceImagesList), skipped)

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

def createVideo(directory, picsPerSecond, videoFPS):
    os.chdir(directory)
    images = os.listdir(directory)
    length = len(os.path.splitext(images[0])[0])
    ff = FFmpeg(
        inputs={},
        outputs={'../out.mp4': '-framerate {0} -i %0{1}d.jpg -c:v libx264 -r {2} -pix_fmt yuv420p'
                .format(picsPerSecond, length, videoFPS)}
    )
    ff.run()
    os.chdir('..')
    filepath = os.path.abspath(os.curdir) + '/out.mp4'
    print 'Your video is complete and located at {0}'.format(filepath)


directory = '/home/kyle/Projects/Photo A Day/Photos/'
subdirectoryName = 'temp/'
subdirectory = directory + subdirectoryName
createTempDir(subdirectory)
images = os.listdir(directory)
faceImagesList = readFaces(images, directory, subdirectory)
resizeAndCrop(faceImagesList)
print 'Check photos in {0} and delete or manually fix any that did not resize properly.'.format(subdirectory)
raw_input("Press Enter to continue...")
renameAsNumbers(subdirectory)
createVideo(subdirectory, 3, 24)

#cleanup
shutil.rmtree(subdirectory)
