import cv2
import sys
import os
import math
import shutil
import exifread
import datetime
from ffmpy import FFmpeg

class Face(object):
    area = 0
    path = ''
    def __init__(self, width, height, path):
        self.area = width * height
        self.path = path

    def scaleFactor(self, desiredArea):
        return math.sqrt(1.0 * desiredArea / self.area)

    def dateTaken(self):
        f = open(self.path, 'rb')
        tags = exifread.process_file(f)
        # TODO add time into the method instead of just chopping it off
        date = datetime.datetime.strptime(str(tags["EXIF DateTimeOriginal"])[:-9], '%Y:%m:%d')
        return date.strftime("%B %d")

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

    newPath = subdirectory + imageName
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

def writeText(image, text):
    origin = (1000, 1000)
    color = (255, 255, 255)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 5, color, 4, cv2.CV_AA)
    return image

def resizeAndCrop():
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
            image = writeText(cropped, face.dateTaken())
            cv2.imwrite(face.path, image)
# necessary for outputting to video
def renameAsNumbers(directory):
    images = os.listdir(directory)
    images.sort()
    numImages = len(images)
    numImagesStrLen = len("{0}".format(numImages))
    counter = 1
    for image in images:
        filetype = image[-4:]
        newName = "{0}".format(counter)
        while (len(newName) < numImagesStrLen):
            newName = "0" + newName
        os.rename(directory + image, "{0}{1}{2}".format(directory, newName, filetype))
        counter += 1

def createVideo(directory):
    os.chdir(directory)
    images = os.listdir(directory)
    length = len(os.path.basename(directory + images[0])[0])
    ff = FFmpeg(
        inputs={},
        outputs={'../out.mp4': '-framerate 3 -i %0{0}d.jpg -c:v libx264 -r 30 -pix_fmt yuv420p'.format(length)}
    )
    ff.run()
    os.chdir('..')
    filepath = os.path.abspath(os.curdir) + '/out.mp4'
    print 'Your video is complete and located at {0}'.format(filepath)

resizeAndCrop()
print 'Check photos in {0} and delete or manually fix any that did not resize properly.'.format(subdirectory)
raw_input("Press Enter to continue...")
renameAsNumbers(subdirectory)
createVideo(subdirectory)

#cleanup
shutil.rmtree(subdirectory)
