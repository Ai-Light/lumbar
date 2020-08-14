#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import matplotlib.pyplot as plt
from mmcv.ops import nms
import numpy as np
import pickle
import json
import SimpleITK as sitk
import pandas as pd
import glob
import os


# In[2]:


with open('../data/DatasetB/test/testB50_series_map.json','rb') as ff:
    test_value=json.load(ff)
    
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


# In[3]:


import glob
import os
import dicomutil
dcm_paths=glob.glob(os.path.join('../data/DatasetB/test/lumbar_testB50', "**", "**.dcm"))
res_=[]
i=0
##循环处理每个dicom
# predictjsompath="./predictions_20200720.json"
tag_list = ['0020|000d','0020|000e','0008|0018','0008|103e','0018|1312']
num=-1
study_list={}
for path in dcm_paths:
    try:
        img_arr = dicom2array(path)
    except:
        continue
    num+=1
    studyUid,seriesUid,instanceUid,seriesDescription,rowcol = dicomutil.dicom_metainfo(path,tag_list)
#     seriesDescription=seriesDescription.upper()
#     if(seriesDescription.find('T2')==-1 and seriesDescription.find('IRFSE')==-1):#没找到T2，不处理。T2WI_SAG.#IRFSE 4000/90/123 5mmS
#         #print('该文件不是T2矢状',path)
#         continue
#     if (seriesDescription.find('TRA') != -1):  #不处理TRA
#         # print('该文件不是T2矢状',path)
#         continue
#         #C4-T2 FSE SAG  STIR, 15 slices.STIR=抑制脂肪序列,T2 IR-TSE
#     if (seriesDescription.find('TRIM') != -1 or seriesDescription.find('IR') != -1 ):  # 不处理TRIM T2_FSE(T),t2_tirm_fs_sag
#         # print('该文件不是T2矢状',path)
#         continue
#     if (seriesDescription.find('(T)') != -1 or seriesDescription.find('AXIAL') != -1):  # 不处理 T2_FSE(T)，T2_FSE_5mm(T)， C5-AXIAL T2 FSE PS, 12 slices
#         # print('该文件不是T2矢状',path)
#         continue
#     #t2_tse_dixon_sag_320_F ，dixon 反应脂肪;:a6OT2 :a6OT2
#     if (seriesDescription.find('DIXON') != -1 or seriesDescription.find('OT2') != -1):  # 不处理TRA
#         # print('该文件不是T2矢状',path)
#         continue
#     #FST2_SAGc时一般有T2WI_SAGc
#     if (seriesDescription.find('FST2') != -1 ):  # 不处理FST2
#         # print('该文件不是T2矢状',path)
#         continue
#     if (rowcol.find('ROW') != -1):  # 不处理ROW
#         # print('该文件不是T2矢状',path)
#         continue
    i = i + 1
    #print('dcm_path:', path)
    if {'studyUid': studyUid,'seriesUid': seriesUid} in test_value:
        if studyUid in study_list:
            if instanceUid+'_'+seriesUid in study_list[studyUid]:
                study_list[studyUid][instanceUid+'_'+seriesUid]['inx'].append(num)
            else:
                study_list[studyUid][instanceUid+'_'+seriesUid]={}
                study_list[studyUid][instanceUid+'_'+seriesUid]['inx']=[]
                study_list[studyUid][instanceUid+'_'+seriesUid]['inx'].append(num)
        else:
            study_list[studyUid]={}
            study_list[studyUid][instanceUid+'_'+seriesUid]={}
            study_list[studyUid][instanceUid+'_'+seriesUid]['inx']=[]
            study_list[studyUid][instanceUid+'_'+seriesUid]['inx'].append(num)
    else:
        pass


# In[4]:


len(study_list)


# In[5]:


label_dict={'T12': 0, 'L1': 1, 'L2': 2, 'L3': 3, 'L4': 4, 'L5': 5, 'T12-L1': 6, 
            'L1-L2': 7, 'L2-L3': 8, 'L3-L4': 9, 'L4-L5': 10, 'L5-S1': 11} 
label_dict={v: k for k, v in label_dict.items()}
tag1_dict={0: 'v2',1: 'v2',2: 'v2',3: 'v2',4: 'v2',5: 'v2',6: 'v1',7: 'v1',8: 'v1',
           9: 'v1',10: 'v1',11: 'v1'}
label_dict


# In[6]:


# v_dict = {'v1-vertebra': 0, 'v2-vertebra': 1, 'v1-disc': 2, 'v2-disc': 3, 'v3-disc': 4, 'v4-disc': 5, 'v5-disc': 6}
# v_dict={v: k for k, v in v_dict.items()}
# v_dict


# In[8]:


result = pickle.load(open("../data/final_result_b.pkl", 'rb'))


# In[9]:


submit_list=[]
sum_count = 0
for k,v in study_list.items():
#     print(v)
    max_list=[]
    num_point=[]
    temp_json={}
    temp_json["studyUid"]=k
    temp_json['data']=[{}]
    temp_json["version"]= "v0.1"
    for k_,v_ in v.items():
        instanceUid,seriesUid=k_.split('_')[0],k_.split('_')[1]
        temp_json2={}
        temp_json2['instanceUid']=instanceUid
        temp_json2['seriesUid']=seriesUid
        temp_json2['annotation']=[]
        for ii in v_['inx']:
            temp_json3={}
            temp_json3['annotator']=54
            temp_json3['data']={}
            temp_json3['data']['point']=[]
            _res=result[ii]
            if len(_res)==0:
                continue
            else:
                res=result[ii]#[0]['keypoints']
            area, score = [],[]
            for _res in res:
                box = _res['bbox'][0]
                area.append(abs(box[0]-box[2])*abs(box[1]-box[3]))
                score.append(box[-1])
            idx = np.argmax(area) #choose the max bbox
            res = [res[idx]]
#             idx = (np.array(score) > 0.3)
#             res = np.array(res)[idx]
            for _res in res:
#                 _res = _res['keypoints']
                _=0
                count = 0
#                 for nn,score in zip(_res['keypoints'][1:], _res['score'][1:]):
                for nn in _res['keypoints'][1:]:
                    _+=1
                    if nn[-1]<0.5:
                        continue
                    else:
                        temp_json4={}
                        x=int(nn[0])
                        y=int(nn[1])
                        temp_json4["coord"]=[x, y]
                        temp_json4["tag"]={}
                        concat_tag=label_dict[_]
#                         v_tag = v_dict[np.argmax(score, axis=-1)]
                        l2 = concat_tag
#                         print(l2, v_tag, score)
                        count += 1
                        if '-' in l2:
                            temp_json4["tag"]['disc']='v1'
#                             if 'disc' not in v_tag:
#                                 temp_json4["tag"]['disc']='v1'
#                             else:
#                                 temp_json4["tag"]['disc']=v_tag.split('-')[0]
                            temp_json4["tag"]['identification']=l2
                            temp_json4['zIndex']=5
                        else:
                            temp_json4["tag"]['vertebra']='v2'
#                             if 'vertebra' not in v_tag:
#                                 temp_json4["tag"]['vertebra']='v2'
#                             else:
#                                 temp_json4["tag"]['vertebra']=v_tag.split('-')[0]                               
                            temp_json4["tag"]['identification']=l2
                            temp_json4['zIndex']=5
                    temp_json3['data']['point'].append(temp_json4)
                sum_count += count
                if len(temp_json3['data']['point'])==0:
                    continue
                else:
                    temp_json2['annotation'].append(temp_json3)
        if len(temp_json2['annotation'])==0:
            continue
        else:
            temp_json['data'][0]=temp_json2
            num_point.append(len(temp_json['data'][0]['annotation'][0]['data']['point']))
            max_list.append(temp_json)
    submit_list.append(max_list[num_point.index(max(num_point))])


# In[10]:


sum_count


# In[11]:


import json
with open('../result/submit_0812_pose_b2_bbox_max.json', 'w') as f_obj:
    json.dump(submit_list, f_obj)


# In[ ]:




