#!/usr/bin/env python
# coding: utf-8

# In[1]:


import SimpleITK as sitk
import pandas as pd
import glob
import os
import json
import cv2
from tqdm import tqdm


# In[2]:


new_dir = '../data/coco/'
if not os.path.exists(new_dir):
    os.mkdir(new_dir)

new_dir = '../data/coco/train2017'
if not os.path.exists(new_dir):
    os.mkdir(new_dir)
    
new_dir = '../data/coco/val2017'
if not os.path.exists(new_dir):
    os.mkdir(new_dir)
    
new_dir = '../data/coco/test2017'
if not os.path.exists(new_dir):
    os.mkdir(new_dir)
    
new_dir = '../data/coco/annotations'
if not os.path.exists(new_dir):
    os.mkdir(new_dir)


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

def get_info(trainPath, jsonPath):  
    annotation_info = pd.DataFrame(columns=('studyUid', 'seriesUid', 'instanceUid', 'annotation'))  
    json_df = pd.read_json(jsonPath)  
    for idx in json_df.index:  
        studyUid = json_df.loc[idx, "studyUid"]  
        seriesUid = json_df.loc[idx, "data"][0]['seriesUid']  
        instanceUid = json_df.loc[idx, "data"][0]['instanceUid']  
        annotation = json_df.loc[idx, "data"][0]['annotation']  
        row = pd.Series(  
            {'studyUid': studyUid, 'seriesUid': seriesUid, 'instanceUid': instanceUid, 'annotation': annotation})  
        annotation_info = annotation_info.append(row, ignore_index=True)  
    dcm_paths = glob.glob(os.path.join(trainPath, "**", "**.dcm"))  # 具体的图片路径  
    # 'studyUid','seriesUid','instanceUid'  
    tag_list = ['0020|000d', '0020|000e', '0008|0018']  
    dcm_info = pd.DataFrame(columns=('dcmPath', 'studyUid', 'seriesUid', 'instanceUid'))  
    for dcm_path in dcm_paths:  
        try:  
            studyUid, seriesUid, instanceUid = dicom_metainfo(dcm_path, tag_list) 
            row = pd.Series(  
                {'dcmPath': dcm_path, 'studyUid': studyUid, 'seriesUid': seriesUid, 'instanceUid': instanceUid})  
            dcm_info = dcm_info.append(row, ignore_index=True)  
        except:  
            pass 
    result = pd.merge(annotation_info, dcm_info, on=['studyUid', 'seriesUid', 'instanceUid'])
    result = result.set_index('dcmPath')['annotation']  # 然后把index设置为路径，值设置为annotation  
    return result


# In[4]:


trainPath = '../data/DatasetA/train/lumbar_train150/'  
jsonPath = '../data/DatasetA/train/lumbar_train150_annotation.json'  
res_train = get_info(trainPath, jsonPath)  # 获取图片路径及对应的annotation
res_train


# In[5]:


trainPath = '../data/DatasetA/train/val/train/'  
jsonPath = '../data/DatasetA/train/lumbar_train51_annotation.json'  
res_val = get_info(trainPath, jsonPath)  # 获取图片路径及对应的annotation


# In[6]:


vertebras = ['v1', 'v2',]
discs = ['v1', 'v2', 'v3', 'v4', 'v5', ]
vertebras_pos = ['vertebra']#['T12', 'L1','L2','L3','L4','L5',]
discs_pos = ['disc']#['T11-T12','T12-L1', 'L1-L2','L2-L3','L3-L4','L4-L5','L5-S1',]
box_label_dict={}
num=0
categories_dict=[]
box_label_dict['box'] = num
categories_dict.append({'id': num, 'name': 'box'})
print(box_label_dict, len(box_label_dict))
box_categories_dict = categories_dict
box_categories_dict


# In[7]:


vertebras = ['v1', 'v2',]
discs = ['v1', 'v2', 'v3', 'v4', 'v5', ]
vertebras_pos = ['T12', 'L1','L2','L3','L4','L5',]
discs_pos = ['T12-L1', 'L1-L2','L2-L3','L3-L4','L4-L5','L5-S1',]
keypoint_label_dict={}
num=0
categories_dict=[]
for p in vertebras_pos:
    label = p
    if label in keypoint_label_dict:
        pass
    else:
        categories_dict.append({'id': num, 'name': label})
        keypoint_label_dict[label]=num
        num+=1
for p in discs_pos:
    label = p
    if label in keypoint_label_dict:
        pass
    else:
        categories_dict.append({'id': num, 'name': label})
        keypoint_label_dict[label]=num
        num+=1
print(keypoint_label_dict, len(keypoint_label_dict))


# In[8]:


key_point_categories_dict = [{"supercategory": "vertebra", 
                    "id": 1, 
                    "name": "vertebra", 
                    "keypoints": ["T12","L1","L2","L3","L4","L5","T12-L1","L1-L2","L2-L3","L3-L4","L4-L5","L5-S1"],
                    "skeleton": [[1,7],[7,2],[2,8],[8,3],[3,9],[9,4],[4,10],[10,5],[5,11],[11,6],[6,12]]
}]
key_point_categories_dict


# # mmp 训练关键点

# In[9]:


label_dict = keypoint_label_dict
res = res_train


# In[10]:


jizhu_json={}
jizhu_json['images']=[]
jizhu_json['annotations']=[]
jizhu_json['categories']=key_point_categories_dict
num_cat=0
image_id=0
anno_id = 0
for ii,jj in zip(res.index,res):
    img_arr = dicom2array(ii)
    save_path='../data/coco/train2017/'+'_'.join(ii.split('/')[-2:]).split('.')[0]+'.jpg'
    cv2.imwrite(save_path,img_arr)
    jizhu_json['images'].append({
        'file_name': '_'.join(ii.split('/')[-2:]).split('.')[0]+'.jpg',
        'height': img_arr.shape[0],
        'width': img_arr.shape[1],
        'id': image_id
    })
    coordxs, coordys, labels = [], [], []
    for n_ in jj[0]['data']['point']:
        w=13
        h=13
        x=n_['coord'][0]
        coordxs.append(x)
        y=n_['coord'][1]
        coordys.append(y)
        
        if(n_['tag']['identification'].find('-')==-1):
            vertebras = n_['tag']['vertebra']
            if (vertebras == ''):  # 没有值，脏数据
                continue
            if(vertebras.find(',')!=-1):#有多个值的时候，先只处理一个
                vertebras=vertebras.split(',')[0]
            else:
                vertebras=[vertebras]
            for vertebra in vertebras:
                label = n_['tag']['identification']#vertebra+'-vertebra'#+n_['tag']['identification']
                if label in label_dict:
                    anno_id += 1
                    labels.append(label)
                else:
                    print(label)
        else:
            discs = n_['tag']['disc']
            if(discs==''):#没有值，脏数据
                continue
            if (discs.find(',')!=-1):#有多个值的时候，先只处理一个
                discs = discs.split(',')
            else:
                discs = [discs]
            for disc in discs:
                label = n_['tag']['identification']#disc+'-disc'#+n_['tag']['identification']
                if label in label_dict:
                    labels.append(label)
                    anno_id += 1
                else:
                    print(label)
    points = [0] * 36
    num_points = 0
    for label, x, y in zip(labels, coordxs, coordys):
        idx = label_dict[label] * 3
        points[idx] = x
        points[idx+1] = y
        points[idx+2] = 2
        num_points += 1

    x = min(coordxs) - 15
    y = min(coordys) - 15
    w = max(coordxs) - x + 30
    h = max(coordys) - y + 30
    jizhu_json['annotations'].append({
                'keypoints':points,
                'num_keypoints': num_points,
                'segmentation': [[]],  # if you have mask labels
                'area': w*h,
                'iscrowd': 0,
                'image_id': image_id,
                'bbox': [x, y, w, h],
                'category_id': 1,
                'id': anno_id,
            })
    anno_id += 1    
    image_id+=1


# In[11]:


with open('../data/coco/annotations/instances_train2017_keypoint.json', 'w') as f_obj:
    json.dump(jizhu_json, f_obj)


# In[12]:


res = res_val


# In[13]:


jizhu_json={}
jizhu_json['images']=[]
jizhu_json['annotations']=[]
jizhu_json['categories']=key_point_categories_dict
num_cat=0
image_id=0
anno_id = 0
for ii,jj in zip(res.index,res):
    img_arr = dicom2array(ii)
    save_path='../data/coco/val2017/'+'_'.join(ii.split('/')[-2:]).split('.')[0]+'.jpg'
    cv2.imwrite(save_path,img_arr)
    jizhu_json['images'].append({
        'file_name': '_'.join(ii.split('/')[-2:]).split('.')[0]+'.jpg',
        'height': img_arr.shape[0],
        'width': img_arr.shape[1],
        'id': image_id
    })
    coordxs, coordys, labels = [], [], []
    for n_ in jj[0]['data']['point']:
        w=13
        h=13
        x=n_['coord'][0]
        coordxs.append(x)
        y=n_['coord'][1]
        coordys.append(y)
        
        if(n_['tag']['identification'].find('-')==-1):
            vertebras = n_['tag']['vertebra']
            if (vertebras == ''):  # 没有值，脏数据
                continue
            if(vertebras.find(',')!=-1):#有多个值的时候，先只处理一个
                vertebras=vertebras.split(',')[0]
            else:
                vertebras=[vertebras]
            for vertebra in vertebras:
                label = n_['tag']['identification']#vertebra+'-vertebra'#+n_['tag']['identification']
                if label in label_dict:
                    anno_id += 1
                    labels.append(label)
                else:
                    print(label)
        else:
            discs = n_['tag']['disc']
            if(discs==''):#没有值，脏数据
                continue
            if (discs.find(',')!=-1):#有多个值的时候，先只处理一个
                discs = discs.split(',')
            else:
                discs = [discs]
            for disc in discs:
                label = n_['tag']['identification']#disc+'-disc'#+n_['tag']['identification']
                if label in label_dict:
                    labels.append(label)
                    anno_id += 1
                else:
                    print(label)
    points = [0] * 36
    num_points = 0
    for label, x, y in zip(labels, coordxs, coordys):
        idx = label_dict[label] * 3
        points[idx] = x
        points[idx+1] = y
        points[idx+2] = 2
        num_points += 1

    x = min(coordxs) - 15
    y = min(coordys) - 15
    w = max(coordxs) - x + 30
    h = max(coordys) - y + 30
    jizhu_json['annotations'].append({
                'keypoints':points,
                'num_keypoints': num_points,
                'segmentation': [[]],  # if you have mask labels
                'area': w*h,
                'iscrowd': 0,
                'image_id': image_id,
                'bbox': [x, y, w, h],
                'category_id': 1,
                'id': anno_id,
            })
    anno_id += 1    
    image_id+=1


# In[14]:


with open('../data/coco/annotations/instances_val2017_keypoint.json', 'w') as f_obj:
    json.dump(jizhu_json, f_obj)


# In[15]:


file_path=glob.glob(os.path.join('../data/DatasetB/test/lumbar_testB50', "**", "**.dcm"))
jizhu_json={}
jizhu_json['images']=[]
jizhu_json['annotations']=[]
jizhu_json['categories']=key_point_categories_dict
image_id=0
anno_id = 0
for ii in tqdm(file_path):
    try:
        img_arr = dicom2array(ii)
    except:
        continue
    save_path='../data/coco/test2017/'+'_'.join(ii.split('/')[-2:]).split('.')[0]+'.jpg'
    cv2.imwrite(save_path,img_arr)
    jizhu_json['images'].append({
        'file_name': '_'.join(ii.split('/')[-2:]).split('.')[0]+'.jpg',
        'height': img_arr.shape[0],
        'width': img_arr.shape[1],
        'id': image_id
    })
    w=13
    h=13

    jizhu_json['annotations'].append({
                'segmentation': [[]],  # if you have mask labels
                'area': w*h,
                'iscrowd': 0,
                'image_id': image_id,
                'bbox': [],
                'category_id': 0,
                'id': anno_id,
            })
    anno_id += 1
    image_id+=1


# In[16]:


with open('../data/coco/annotations/instances_test2017B_keypoint.json', 'w') as f_obj:
    json.dump(jizhu_json, f_obj)


# # mmd 训练大框

# In[17]:


res = res_train


# In[18]:


jizhu_json={}
jizhu_json['images']=[]
jizhu_json['annotations']=[]
jizhu_json['categories']=box_categories_dict
num_cat=0
image_id=0
anno_id = 0
for ii,jj in zip(res.index,res):
    img_arr = dicom2array(ii)

    jizhu_json['images'].append({
        'file_name': '_'.join(ii.split('/')[-2:]).split('.')[0]+'.jpg',
        'height': img_arr.shape[0],
        'width': img_arr.shape[1],
        'id': image_id
    })
    coordxs, coordys, labels = [], [], []
    for n_ in jj[0]['data']['point']:
        w=13
        h=13
        x=n_['coord'][0]
        coordxs.append(x)
        y=n_['coord'][1]
        coordys.append(y)

    points = [0] * 36
    num_points = 0

    x = min(coordxs) - 15
    y = min(coordys) - 15
    w = max(coordxs) - x + 30
    h = max(coordys) - y + 30
    jizhu_json['annotations'].append({
                'keypoints':points,
                'num_keypoints': num_points,
                'segmentation': [[]],  # if you have mask labels
                'area': w*h,
                'iscrowd': 0,
                'image_id': image_id,
                'bbox': [x, y, w, h],
                'category_id': 0,
                'id': anno_id,
            })
    anno_id += 1    
    image_id+=1


# In[19]:


with open('../data/coco/annotations/instances_train2017_bbox.json', 'w') as f_obj:
    json.dump(jizhu_json, f_obj)


# In[20]:


res = res_val


# In[21]:


jizhu_json={}
jizhu_json['images']=[]
jizhu_json['annotations']=[]
jizhu_json['categories']=box_categories_dict
num_cat=0
image_id=0
anno_id = 0
for ii,jj in zip(res.index,res):
    img_arr = dicom2array(ii)

    jizhu_json['images'].append({
        'file_name': '_'.join(ii.split('/')[-2:]).split('.')[0]+'.jpg',
        'height': img_arr.shape[0],
        'width': img_arr.shape[1],
        'id': image_id
    })
    coordxs, coordys, labels = [], [], []
    for n_ in jj[0]['data']['point']:
        w=13
        h=13
        x=n_['coord'][0]-6
        coordxs.append(x+6)
        y=n_['coord'][1]-6
        coordys.append(y+6)

    points = [0] * 36

    x = min(coordxs) - 15
    y = min(coordys) - 15
    w = max(coordxs) - x + 30
    h = max(coordys) - y + 30
    jizhu_json['annotations'].append({
                'keypoints':points,
                'num_keypoints': num_points,
                'segmentation': [[]],  # if you have mask labels
                'area': w*h,
                'iscrowd': 0,
                'image_id': image_id,
                'bbox': [x, y, w, h],
                'category_id': 0,
                'id': anno_id,
            })
    anno_id += 1    
    image_id+=1


# In[22]:


import json
with open('../data/coco/annotations/instances_val2017_bbox.json', 'w') as f_obj:
    json.dump(jizhu_json, f_obj)


# In[23]:


file_path=glob.glob(os.path.join('../data/DatasetB/test/lumbar_testB50', "**", "**.dcm"))
jizhu_json={}
jizhu_json['images']=[]
jizhu_json['annotations']=[]
jizhu_json['categories']=box_categories_dict
image_id=0
anno_id = 0
for ii in tqdm(file_path):
    try:
        img_arr = dicom2array(ii)
    except:
        continue
#     save_path='../coco_data/test2017_b/'+'_'.join(ii.split('/')[-2:]).split('.')[0]+'.jpg'
#     cv2.imwrite(save_path,img_arr)
    jizhu_json['images'].append({
        'file_name': '_'.join(ii.split('/')[-2:]).split('.')[0]+'.jpg',
        'height': img_arr.shape[0],
        'width': img_arr.shape[1],
        'id': image_id
    })
    w=13
    h=13

    jizhu_json['annotations'].append({
                'segmentation': [[]],  # if you have mask labels
                'area': w*h,
                'iscrowd': 0,
                'image_id': image_id,
                'bbox': [],
                'category_id': 0,
                'id': anno_id,
            })
    anno_id += 1
    image_id+=1


# In[24]:


with open('../data/coco/annotations/instances_test2017B_bbox.json', 'w') as f_obj:
    json.dump(jizhu_json, f_obj)

