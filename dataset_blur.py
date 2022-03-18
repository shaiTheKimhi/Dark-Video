import os
import random
import torch
import cv2
import numpy as np

def getDataset(set_mode = 'train'):
    ds = []
    ds_list = []

    root_dir = os.path.join(os.getcwd(),os.path.join('dataset','GOPRO_Large'))
    set_dir = os.path.join(root_dir,set_mode)

    data_dirs_list = [x for x in os.walk(set_dir)][0][1]
    data_dirs_frames_list = [x.split("_") for x in data_dirs_list]
    data_dirs_frames_list = [int(x[1]) for x in data_dirs_frames_list]
    data_dirs_list = [os.path.join(set_dir,x) for x in data_dirs_list]

    if set_mode == 'train':
        for data_dir, data_frames in zip(data_dirs_list[:1],data_dirs_frames_list[:1]):
            sharp_dir = os.path.join(data_dir,'sharp')
            images_list = [x for x in os.walk(sharp_dir)]
            images_list = images_list[0][2]
            gt_list = images_list[int((data_frames-1)/2):-int((data_frames-1)/2)]
    
            for i, gt in enumerate(gt_list):
                gt_blured_frames = images_list[i:i+data_frames]
                frame1, frame2 = random.sample(gt_blured_frames,2) # random peak of 2 frames


                gtimage = torch.from_numpy(cv2.imread(os.path.join(sharp_dir, gt))).permute(-1,0,1)
                image1 = torch.from_numpy(cv2.imread(os.path.join(sharp_dir, frame1))).permute(-1,0,1)
                image2 = torch.from_numpy(cv2.imread(os.path.join(sharp_dir, frame2))).permute(-1,0,1)
                new_train_data_list = [frame1, frame2, gt]
                new_train_data_list = [os.path.join(sharp_dir, x) for x in new_train_data_list]
                new_train_data = [image1, image2, gtimage]
                new_train_data = [x.unsqueeze(0) for x in new_train_data]
                new_train_data = [x.float() for x in new_train_data]
                ds_list.append(new_train_data_list)
                ds.append(new_train_data)
                a = os.getcwd()

    else: # test mode
        for data_dir in data_dirs_list[:1]:
            sharp_dir = os.path.join(data_dir,'sharp')
            blur_dir = os.path.join(data_dir,'blur')
            images_list = [x for x in os.walk(sharp_dir)]
            images_list = images_list[0][2]
            blur_images_list = [x for x in os.walk(blur_dir)]
            blur_images_list = blur_images_list[0][2]
    
            for gt, blur in zip(images_list, blur_images_list):
              
                gtimage = torch.from_numpy(cv2.imread(os.path.join(sharp_dir, gt))).permute(-1,0,1)
                blurimage = torch.from_numpy(cv2.imread(os.path.join(blur_dir, blur))).permute(-1,0,1)
                new_test_data_list = [os.path.join(blur_dir, blur), os.path.join(sharp_dir, gt)]
                new_test_data = [blurimage, gtimage]
                new_test_data = [x.unsqueeze(0) for x in new_test_data]
                new_test_data = [x.float() for x in new_test_data]
                ds_list.append(new_test_data_list)
                ds.append(new_test_data)

    return ds, ds_list


getDataset() 