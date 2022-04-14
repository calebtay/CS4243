import os
import numpy as np
import cv2
import glob
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torch
from torchvision import transforms as trans

test_data_percentage = 0.1
validation_data_percentage = 0.1

output_size = 200
resize_resolution = 256

folders = os.listdir(os.path.dirname(os.path.realpath(__file__)) + '/Images/')
#print(str(os.path.dirname(os.path.realpath(__file__)) + '/Images/'))

train_data = []
label_data = []
test_data = []
test_label_data = []
valid_data = []
valid_label_data = []

label_dict = {}

label_index = 0
total_images = 0
for folder in folders:
    label = os.path.basename(folder)
    #print(label)
    images_path = glob.glob(os.path.dirname(os.path.realpath(__file__)) + '/Images/' + label + "/*.jpg")

    if label not in label_dict.keys():
        label_dict.update({label: label_index})

    image_index = 0
    total_images = total_images + len(images_path)
    test_data_count = int(len(images_path) * test_data_percentage)
    validation_data_count = int(len(images_path) * validation_data_percentage)
    #print(test_data_count)

    for img in images_path: #running a loop to iterate through every image in the file
        pic = Image.open(img)

        width, height = pic.size
        
        if (width > height):
            boundary = (width - height) / 2
            pic = pic.crop((0, -boundary, width, height + boundary))

        imgSmall = pic.resize((resize_resolution,resize_resolution),resample=Image.BILINEAR)
        # Scale back up using NEAREST to original size
        result = imgSmall.resize([output_size,output_size],Image.NEAREST)
        #result.show()
        #print(label_index)
        result = trans.functional.pil_to_tensor(result)

        if (result.size()[0] == 3):
            #print(result.size())
            if (image_index < test_data_count):
                test_data.append(result)
                test_label_data.append(torch.tensor([label_index]))
            elif (image_index >= test_data_count and image_index < validation_data_count + test_data_count):
                valid_data.append(result)
                valid_label_data.append(torch.tensor([label_index]))
            else:
                train_data.append(result)
                label_data.append(torch.tensor([label_index]))

            image_index = image_index + 1
        #break
    label_index = label_index + 1
    print(str(total_images) + " imgaes completed. " + label + " folder completed.")   
    #break

print("Total images: " + str(total_images))
print("Train data length: " + str(len(train_data)))
print("Train label length: " + str(len(label_data)))
print("Test data length: " + str(len(test_data)))
print("Test label length: " + str(len(test_label_data)))
print("Validation data length: " + str(len(valid_data)))
print("Validation label length: " + str(len(valid_label_data)))


train_data = torch.stack(train_data)
label_data = torch.stack(label_data)
test_data = torch.stack(test_data)
test_label_data = torch.stack(test_label_data)
valid_data = torch.stack(valid_data)
valid_label_data = torch.stack(valid_label_data)

torch.save(train_data, os.path.dirname(os.path.realpath(__file__)) + '/train_data.pt')
torch.save(label_data, os.path.dirname(os.path.realpath(__file__)) + '/label_data.pt')
torch.save(test_data, os.path.dirname(os.path.realpath(__file__)) + '/test_data.pt')
torch.save(test_label_data, os.path.dirname(os.path.realpath(__file__)) + '/test_label.pt')
torch.save(valid_data, os.path.dirname(os.path.realpath(__file__)) + '/valid_data.pt')
torch.save(valid_label_data, os.path.dirname(os.path.realpath(__file__)) + '/valid_label.pt')

"""
print(type(train_data))
print(type(label_data))
print(type(test_data))
print(type(test_label_data))
"""