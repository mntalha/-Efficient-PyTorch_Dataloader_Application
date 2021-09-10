# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 14:03:58 2021

@author: talha
"""


#

import os 
from torch.utils.data import DataLoader


#define img_path 
img_path = "../3-Data/train_img"
import timeit


#check path control
if isinstance(img_path, str): #Image path is a string 
    if os.path.exists(img_path):
        print ("Path is exist")
    else:
        raise FileNotFoundError('Path is not found on your system')
        
        
        

#list the image_name
imgs_name = os.listdir(img_path) #Read the image name inside the path
img_array = []



import cv2
for img in imgs_name:    
    img_array.append(cv2.imread(os.path.join(img_path, img)))   



if __name__ == '__main__':
    
    #############################
    #number of worker > 0 , it uses paralel programming and main is required.
    train_dataloader = DataLoader(img_array, batch_size=64,num_workers=0,shuffle=False ,pin_memory=False)
    
    start = timeit.default_timer()
    for i in train_dataloader:
        # print(i)
        print("The time difference is :", timeit.default_timer() - start)
        break
    #The time difference is : 0.0006987999999998884
    #2 worker -The time difference is : 3.417489599999996
    #4 worker -The time difference is :  6.861533100000003
    #12 worker -The time difference is :  20.43391109999999
    # #############################

    from torchvision.datasets import ImageFolder
    from torchvision.transforms import ToTensor


    train_datasett = ImageFolder("../3-Data",transform=(ToTensor()))
    
    train_dataloaderr = DataLoader(train_datasett, batch_size=64,num_workers=0,shuffle=False ,pin_memory=False)
    
    start = timeit.default_timer()
    
    for i in train_dataloaderr:
        # print(i)
        print("The time difference is :", timeit.default_timer() - start)
        break
    #The time difference is : 0.02038159999999989
    #2 worker -The time difference is : 3.2852503000000013
    #4 worker -The time difference is : 6.720856500000025
    #12 worker -The time difference is :  20.35810570000001
    # #############################
    from torchvision import datasets

    training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
    )
    train_dataloader = DataLoader(training_data, batch_size=64,num_workers=0,shuffle=False ,pin_memory=False)
    start = timeit.default_timer()
    
    for i in train_dataloaderr:
        # print(i)
        print("The time difference is :", timeit.default_timer() - start)
        break
    #The time difference is : 0.01748929999999982
    #2 worker -The time difference is : 3.3209216000000055
    #4 worker -The time difference is : 6.843560699999983
    #12 worker -The time difference is :  20.102706399999988
    # #############################
    import sys
    sys.path.append('../5-Dataset_Class/')
    from custom_dataset import CustomImageDataset
    
    dataset=CustomImageDataset(img_path)  
    trainloader = DataLoader(dataset=dataset,batch_size=64,num_workers=12, shuffle=False,pin_memory=False)
    
    start = timeit.default_timer()
    for i in trainloader:
        # print(i)
        print("The time difference is :", timeit.default_timer() - start)
        break
    #The time difference is :  0.022718800000001593
    #2 worker -The time difference is : 3.4220702
    #4 worker -The time difference is : 6.8258472999999995
    #12 worker -The time difference is :  20.3560176
    # #############################

