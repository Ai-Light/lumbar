#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from argparse import ArgumentParser
import numpy as np
from mmpose.apis import inference_pose_model, init_pose_model, vis_pose_result
import matplotlib.pyplot as plt
import mmcv
from mmpose.datasets import build_dataloader, build_dataset
from mmcv import Config, DictAction
import glob
import SimpleITK as sitk
import pandas as pd
import glob
from tqdm import tqdm
import pickle


# In[2]:


pose_config = './mmpose/configs/top_down/hrnet/coco/hrnet_w32_coco_384x288.py'
pose_checkpoint = './mmpose/work_dirs/0/epoch_205.pth'

# build the pose model from a config file and a checkpoint file
pose_model = init_pose_model(pose_config, pose_checkpoint, device='cuda:0')


# In[3]:


def dicom_metainfo(dicm_path, list_tag):
    '''
    获取dicom的元数据信息
    :param dicm_path: dicom文件地址
    :param list_tag: 标记名称列表,比如['0008|0018',]
    :return:
    '''
    reader = sitk.ImageFileReader()
    reader.LoadPrivateTagsOn()
    reader.SetFileName(dicm_path)
    reader.ReadImageInformation()
    return [reader.GetMetaData(t) for t in list_tag]


def dicom2array(dcm_path):
    '''
    读取dicom文件并把其转化为灰度图(np.array)
    https://simpleitk.readthedocs.io/en/master/link_DicomConvert_docs.html
    :param dcm_path: dicom文件
    :return:
    '''
    image_file_reader = sitk.ImageFileReader()
    image_file_reader.SetImageIO('GDCMImageIO')
    image_file_reader.SetFileName(dcm_path)
    image_file_reader.ReadImageInformation()
    image = image_file_reader.Execute()
    if image.GetNumberOfComponentsPerPixel() == 1:
        image = sitk.RescaleIntensity(image, 0, 255)
        if image_file_reader.GetMetaData('0028|0004').strip() == 'MONOCHROME1':
            image = sitk.InvertIntensity(image, maximum=255)
        image = sitk.Cast(image, sitk.sitkUInt8)
    img_x = sitk.GetArrayFromImage(image)[0]
    return img_x

dcm_paths=glob.glob(os.path.join('../data/DatasetB/test/lumbar_testB50', "**", "**.dcm"))
files=[]
for path in dcm_paths:
    try:
        img_arr = dicom2array(path)
    except:
        continue
    files.append(path)
paths = []
for f in files:
    paths.append('../data/coco/test2017/' + '_'.join(f.split('/')[-2:]).split('.')[0]+'.jpg')
len(paths)


# In[4]:


result = pickle.load(open("../data/results_bbox_test.pkl", 'rb'))
len(result)


# In[5]:


final_result = []
for idx, (img, r) in tqdm(enumerate(zip(paths, result))):
    
    person_bboxes = r[-1]#[[128.54112   ,  64.24658   , 143.41219   , 156.22845   ,0.98622096]]

    pose_results = inference_pose_model(
            pose_model, img, person_bboxes, format='xyxy')
    
    final_result.append(pose_results)

#     palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
#                         [230, 230, 0], [255, 153, 255], [153, 204, 255],
#                         [255, 102, 255], [255, 51, 255], [102, 178, 255],
#                         [51, 153, 255], [255, 153, 153], [255, 102, 102],
#                         [255, 51, 51], [153, 255, 153], [102, 255, 102],
#                         [51, 255, 51], [0, 255, 0]])

#     pose_limb_color = palette[[
#         0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
#     ]]
#     pose_kpt_color = palette[[
#         16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
#     ]]

#     img = pose_model.show_result(
#         img,
#         pose_results,
#         None,
#         pose_kpt_color=pose_kpt_color,
#         pose_limb_color=pose_limb_color,
#         kpt_score_thr=0.3,
#         show=False,
#         out_file=None)
#     plt.figure(figsize=(15, 10))
#     plt.imshow(mmcv.bgr2rgb(img))
#     plt.show()
    
len(final_result)


# In[6]:


with open('../data/final_result_b.pkl', 'wb') as outp:
    pickle.dump(final_result, outp)


# In[10]:





