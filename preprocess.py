# Importing Image class from PIL module
from PIL import Image
import os

training_path = "D:/Eamon/Documents/Coding/Python/ScienceFair/imgDataset/"
dirs = [directory for directory in os.listdir(training_path)]

def process_images(dirs, im_path):
    for directory in dirs:
        for file in os.listdir(im_path + directory): 
            # Opens a image in RGB mode
            im = Image.open(im_path + directory + "/" + file)
            im = im.resize((180, 180))
            print(im.size)
            im.save(im_path + directory + "/" + file)

#process_images(dirs, training_path)