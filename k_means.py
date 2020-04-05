'''
Edit by donser
K_means_ImageCluster
K-means for image compressing using：python/opencv
'''
import numpy as np
import random
import cv2 as cv
from tqdm import tqdm
import sys
from sys import argv
#输入图像名 K数 迭代次数 输出文件名
py , input_img , K , iterator , outout_img = argv

def _img_to_list(img):
    '''
    numpy中的h*w*c，转为[[r,g,b],[r,g,b]...]形式
    输入：numpy图像
    输出：numpy列表
    '''
    img=img.flatten()
    img=np.reshape(img,[len(img)//3,3])
    return img

def _init_centroid(img_list,K):
    '''
    初始化K个质心点，为了加快收敛找像素最值均匀分配K份
    输入：img_list 像素值rgb
    输入：K 质心数量
    输出：K个初始化的质心rgb值
    '''
    centroid=[]
    record_new=[0.,0.,0.]
    record_max=[0.,0.,0.]
    record_min=[255.0,255.0,255.0]
    for i in range(img_list.shape[0]):
        cmp=img_list[i][0]+img_list[i][1]+img_list[i][2]
        max=record_max[0]+record_max[1]+record_max[2]
        min=record_min[0]+record_min[1]+record_min[2]
        if cmp>max:
            record_max=img_list[i]
        if cmp<min:
            record_min=img_list[i]
    
    for i in range(K):
        record_new=((record_max-record_min)/(K-1))*i+record_min
        centroid.append(record_new)
    return np.asarray(centroid)

def _judgement_centroid(img_list,centroid,last_or_not):
    '''
    判断每个像素点所属的类别
    输入：img_list 图像的像素值rgb
    输入：当前K个质心像素值rgb
    输入：last_or_not 是否是最后一次更新类别(0非最后 1最后)
    输出：每个像素点所属的类别
    输出：ave 仅对最后一次更新类别有效，输出每个类别的rgb均值
    '''
    classify=[]
    ave=np.zeros([K,3],dtype=np.float64)
    count=np.zeros([K,1],dtype=np.int64)
    for i in range(img_list.shape[0]):
        bias=img_list[i]-centroid
        dis=np.multiply(bias,bias).sum(axis=1)
        classify.append(dis.argmin())
        if last_or_not==1:
            ave[dis.argmin()]+=img_list[i]
            count[dis.argmin()]+=1
    for i in range (K):
        if count[i]==0:
            count[i]=1
    return np.asarray(classify),ave/count

def _update_centroid(img_list,classify,K):
    '''
    根据类别更新K个质心点rgb
    输入：img_list 图像的像素值rgb
    输入：classify 每个像素点所属的类别
    输出：新的质心点的rgb
    '''
    centroid=np.zeros([K,3],dtype=np.float64)
    count=np.zeros([K,1],dtype=np.int64)
    for i in range(img_list.shape[0]):
        j=int(classify[i])
        centroid[j]=np.add(centroid[j],img_list[i])
        count[j]+=1
    for i in range (K):
        if count[i]==0:
            count[i]=1
    return centroid/count

def _representative_class(classify,ave):
    '''
    计算每个类别的代表元素
    输入：classify 每个像素点所属的类别
    输出：img_out新的rgb值
    '''
    img_out=np.zeros([classify.shape[0],3],dtype=np.int64)
    for i in range(classify.shape[0]):
        img_out[i]=ave[classify[i]]
    return img_out

def _make_output_img(img_out,outout_img):
    '''
    输入：img_out新的rgb值
    输入：图像的名称
    '''
    img_out=np.reshape(img_out,[h,w,c])
    cv.imwrite(outout_img,img_out)



#读入图像 统一转为3通道
img = cv.imread(input_img)
try:
    img.shape
except:
    print("Faile to load:",input_img)
    sys.exit(0)
img = img.astype(np.float64)
h = img.shape[0]
w = img.shape[1]
c = img.shape[2]
K=int(K)
iterator=int(iterator)
print("Parameter:")
print("Input Height:",h," Width:",w," Deepth:",c)
print("K-size:",K," iterator:",iterator)

#格式转换
img_list = _img_to_list(img)
#初始化K个质心点rgb
centroid = _init_centroid(img_list,K)

#进度条
pbar = tqdm(total=iterator)
#循环更新iterator次
for i in range(iterator):
    pbar.update(1)
    #判断每个像素点所属的类别
    classify,_=_judgement_centroid(img_list,centroid,0)
    #根据类别更新K个质心点rgb
    centroid=_update_centroid(img_list,classify,K)
pbar.close()
    
#最后一次更新每个像素点所属的类别   
classify,ave=_judgement_centroid(img_list,centroid,1)

#计算每个类别的代表元素
img_out=_representative_class(classify,ave)

#格式转换 & 成图输出
_make_output_img(img_out,outout_img)
print("Done ")



