# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 14:03:58 2021

@author: talha
"""

import os 
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import timeit


class CustomImageDataset(Dataset):
    
    def __init__(self, img_path,transform =True):
        
        self.img_path = img_path
        
        self.transform = transform
        
        if self.check_path(self.img_path):
            self.img_name_list = os.listdir(self.img_path)
            
        self.transform_operation = transforms.Compose([
            
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            ])
    
    def check_path(self,path):
        #check path control
        if isinstance(path, str): #Image path is a string
            if os.path.exists(path):
                #☺print ("Path is exist")
                return True
            else:
                raise FileNotFoundError('Path is not found on your system')  
    ##you can control with len(object)
    def __len__(self):
        return len(self.img_name_list)
    
    def __getitem__(self, idx):
        
        img_name =os.path.join(self.img_path,self.img_name_list[idx])
        
        #cv2 could be used but pillow has  straightforward structure
        image = Image.open(img_name).convert("L") #gray scale image
        
        if self.transform:
            image = self.transform_operation(image)
            
       
        return (image)

        
if __name__ == "__main__":
    
    img_path = "../3-Data/train_img"
    dataset=CustomImageDataset(img_path)  
    trainloader = DataLoader(dataset=dataset,batch_size=64,num_workers=0, shuffle=False,pin_memory=False)
    
    start = timeit.default_timer()
    for i in trainloader:
        # print(i)
        print("The time difference is :", timeit.default_timer() - start)
        break

