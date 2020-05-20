import cv2
import numpy as np
import os
from os.path import isfile, join
from natsort import natsorted
import moviepy.editor as mp
import warnings
import numba
from agec import *

warnings.filterwarnings("ignore")
file_name = "input.mp4" #input('Enter Input File Name: ')
nightmode = True #"y" in input("RTX needed? ").lower()
vidcap = cv2.VideoCapture(file_name)
fps = vidcap.get(cv2.CAP_PROP_FPS)
duration = vidcap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

@numba.jit
def AISRE(image):
    if nightmode:
        image = agec_init(image)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("imgfolder/image" + str(count) + ".png", image)  # save frame as PNG uncompressed file

@numba.jit
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = vidcap.read()
    print(count)
    if hasFrames:
        AISRE(image)
    return hasFrames


sec = 0
frameTime = 1 / fps  # //fps = 1/framerate, framerate = 1/fps
count = 1
success = getFrame(sec)

# takes snapshot of video at every value of sec, 1 pic every frameTime secs
# for 2fps, 1 pic every 0.5secs
while success:
    count = count + 1
    sec = sec + frameTime
    sec = round(sec, 4)
    success = getFrame(sec)

file_name_without_extension = file_name.split('.')[0]
pathIn = './imgfolder/'
pathOut = './vidfolder/output_' + file_name_without_extension + '.avi'
frame_array = []

files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
# for sorting the file names properly
files = natsorted(files)

img_tmp = cv2.imread(pathIn + files[0])
height, width, layers = img_tmp.shape
size = (width, height)

# combines all pics to get video with 'fps' being output video's FPS
out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(files)):
    # reading each files
    print(pathIn + files[i])
    img = cv2.imread(pathIn + files[i])
    # inserting the frames into an image array
    out.write(img)

out.release()

for file in files:
    os.remove(pathIn + file)

print("FPS of input video: ", fps)
print("Length of input video: ", duration, "secs")

vidcap = cv2.VideoCapture('./vidfolder/output_' + file_name_without_extension + '.avi')
fps = vidcap.get(cv2.CAP_PROP_FPS)
duration = vidcap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

print("FPS of output video: ", fps)
print("Length of output video: ", duration, "secs")

videoIn = mp.VideoFileClip(file_name)
videoOut = mp.VideoFileClip('./vidfolder/output_' + file_name_without_extension + '.avi')
audioIn = videoIn.audio

audioFileName = file_name_without_extension + ".mp3"
audioIn.write_audiofile(audioFileName)

videoOut.write_videofile('./vidfolder/output_' + file_name_without_extension + '.mp4', audio=audioFileName)
os.remove(audioFileName)
os.remove('./vidfolder/output_' + file_name_without_extension + '.avi')