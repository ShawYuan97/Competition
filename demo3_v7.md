## 导库


```python
#导入python基本库
import hashlib
import os
import time
from glob import glob
import random

#导入科学计算库
import numpy as np
import pandas as pd

#导入处理图片库
import cv2

#导入打开图片的库
import PIL
from PIL import Image
from PIL import ImageStat

#导入进度条库
from tqdm.notebook import tqdm

#导入画图的库
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

#导入深度学习库
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler  #导入混合精度计算库
from torch.nn import functional as F

#导入sk-learn
import sklearn
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold

#导入图片增强库
import albumentations
from albumentations import (HorizontalFlip, VerticalFlip, Rotate,
                            ShiftScaleRotate, RandomBrightnessContrast,
                            Perspective, CLAHE, Transpose, Blur,
                            OpticalDistortion, GridDistortion,
                            HueSaturationValue, ColorJitter, GaussNoise,
                            MotionBlur, MedianBlur, Emboss, Sharpen, Flip,
                            OneOf, SomeOf, Compose, Normalize, CoarseDropout,
                            CenterCrop, GridDropout, Resize)  #导入图片增强的一些变换
from albumentations.pytorch import ToTensorV2

#导入深度学习预训练库
import timm

#导入torchvision
import torchvision

#导入d2l
import d2l

#导入深度学习辅助工具
from torchinfo import summary  #查看模型框架

#忽略ignore
import warnings

warnings.filterwarnings('ignore')
```

## 定义超参数


```python
CFG = {
    'height': 224,
    'width': 224,
    'train_root': '../dataset/competition_dataset/facial_dataset/train',
    'test_root': '../dataset/competition_dataset/facial_dataset/test',
    'seed': 123,
    'model_arch': 'ResNeSt50',
    'epochs': 15,
    'train_bs': 256,
    'valid_bs': 256,
    'T_0': 16,  # Number of iterations for the first restart
    'lr': 8e-4,
    'min_lr': 1e-7,
    'weight_decay': 5e-4,
    'num_workers': 22,
    'accum_iter': 2,  # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,  # 0 (silent),1 (progress bar), 2 (one line per epoch)
    #     'device': 'cpu'
    'device': 'cuda:0',
    'file_name': 'demo3_v7_ResNeSt50',
    'class_nums': 7,
    'fold_num': 21,
} 
```

### scheduler研究


```python
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR
import torch.nn as nn
from torchvision.models import resnet18
import matplotlib.pyplot as plt
#
model = resnet18(pretrained=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
mode = 'cosineAnnWarm'
if mode == 'cosineAnn':
    scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
elif mode == 'cosineAnnWarm':
    scheduler = CosineAnnealingWarmRestarts(optimizer,
                                            T_0=21,
                                            T_mult=2,
                                            eta_min=1e-6)
    '''
    以T_0=5, T_mult=1为例:
    T_0:学习率第一次回到初始值的epoch位置.
    T_mult:这个控制了学习率回升的速度
        - 如果T_mult=1,则学习率在T_0,2*T_0,3*T_0,....,i*T_0,....处回到最大值(初始学习率)
            - 5,10,15,20,25,.......处回到最大值
        - 如果T_mult>1,则学习率在T_0,(1+T_mult)*T_0,(1+T_mult+T_mult**2)*T_0,.....,(1+T_mult+T_mult**2+...+T_0**i)*T0,处回到最大值
            - 5,15,35,75,155,.......处回到最大值
    example:
        T_0=5, T_mult=1
    '''
plt.figure()
max_epoch = 20
iters = 200
cur_lr_list = []
for epoch in range(max_epoch):
    for batch in range(iters):
        '''
        这里scheduler.step(epoch + batch / iters)的理解如下,如果是一个epoch结束后再.step
        那么一个epoch内所有batch使用的都是同一个学习率,为了使得不同batch也使用不同的学习率
        则可以在这里进行.step
        '''
        #scheduler.step(epoch + batch / iters)
        optimizer.step()
    scheduler.step()
    cur_lr = optimizer.param_groups[-1]['lr']
    cur_lr_list.append(cur_lr)
#     print('cur_lr:',cur_lr)
x_list = list(range(len(cur_lr_list)))
plt.plot(x_list, cur_lr_list)
plt.show()
```

## 数据导入

### 文件目录


```python
# -- competition
#   -- facial expression recognition.py
# -- dataset
#     -- competition_dataset
#         -- facial_dataset
#             -- sample_submit.csv
#             -- test
#                 -- 00001.png
#                 -- ...
#                 -- 07178.png
#             -- train
#                 -- angry
#                     -- im0.png
#                     -- ...
#                 -- disgusted
#                     -- im0.png
#                     -- ...
#                 -- fearful
#                     -- im0.png
#                     -- ...
#                 -- happy
#                     -- im0.png
#                     -- ...
#                 -- neutral
#                     -- im0.png
#                     -- ...
#                 -- sad
#                     -- im0.png
#                     -- ...
#                 -- surprised
#                     -- im0.png
#                     -- ...
```

### 数据展示


```python
#数据展示
train_data = torchvision.datasets.ImageFolder(
    root='../dataset/competition_dataset/facial_dataset/train')

class_to_idx = train_data.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
# ncols, nrows = 4, 6
# fig, axes = plt.subplots(nrows=nrows,
#                          ncols=ncols,
#                          figsize=(12, 12),
#                          subplot_kw={
#                              'xticks': [],
#                              'yticks': []
#                          })
# for row in range(nrows):
#     for col in range(ncols):
#         img = train_data[np.random.randint(len(train_data))]
#         while img[1] != row:
#             img = train_data[np.random.randint(len(train_data))]
#         axes[row][col].imshow(img[0])
#         if col == 0:
#             axes[row][col].set_ylabel(idx_to_class[img[1]], fontsize=20)
# plt.tight_layout()
CFG['class_to_idx'] = train_data.class_to_idx
CFG['idx_to_class'] = {v: k for k, v in CFG['class_to_idx'].items()}
```

### 数据清洗


```python
#获得每张图片的hash值
import hashlib


def get_hash(image):
    md5 = hashlib.md5()
    md5.update(np.asarray(image).tobytes())  #利用np.asarray比np.array更快
    return md5.hexdigest()  #返回十六进制的哈希值


#获取验证集图片和方差
import os
from PIL import Image
import PIL
import hashlib
import numpy as np
import pandas as pd


def get_hash(image):
    md5 = hashlib.md5()
    md5.update(np.asarray(image).tobytes())  #利用np.asarray比np.array更快
    return md5.hexdigest()  #返回十六进制的哈希值


def compute_stat_of_img(path, mode='test'):
    if mode == 'train':
        train_data = torchvision.datasets.ImageFolder(root=path)
        imgnames = [cont[0].split('train/')[1] for cont in train_data.imgs]
        labels = [sample[1] for sample in train_data]
    else:
        imgnames = sorted(os.listdir(path))
    meta_test = []
    for i, imgsingle in enumerate(imgnames):
        img = Image.open(os.path.join(path, imgsingle))
        extrema = img.getextrema()
        stat = PIL.ImageStat.Stat(img)
        if len(stat.mean) == 3:
            meta = {
                'image': imgsingle,
                'hash': get_hash(img),
                'R_min': extrema[0][0],
                'R_max': extrema[0][1],
                'G_min': extrema[1][0],
                'G_max': extrema[1][1],
                'B_min': extrema[2][0],
                'B_max': extrema[2][1],
                'R_avg': stat.mean[0],
                'G_avg': stat.mean[1],
                'B_avg': stat.mean[2],
                'height': img.height,
                'width': img.width,
                'format': img.format,
                'mode': img.mode
            }
        elif len(stat.mean) == 1:
            meta = {
                'image': imgsingle,
                'hash': get_hash(img),
                'R_min': extrema[0],
                'R_max': extrema[1],
                'R_avg': stat.mean[0],
                'std': stat.stddev[0],
                'format': img.format,
                'mode': img.mode
            }
        if mode == 'train':
            meta['label_name'] = CFG['idx_to_class'][labels[i]]
            meta['label'] = labels[i]
        meta_test.append(meta)
    return pd.DataFrame(meta_test)
```


```python
df_test = compute_stat_of_img(
    '../dataset/competition_dataset/facial_dataset/test', mode='test')
print('test dataset std : {:.4f} mean: {:.4f}'.format(
    (df_test['std'] / (df_test.R_max - df_test.R_min)).mean(),
    (df_test.R_avg / (df_test.R_max - df_test.R_min)).mean()))
std = (df_test['std'] / (df_test.R_max - df_test.R_min)).mean()
mean = (df_test.R_avg / (df_test.R_max - df_test.R_min)).mean()
CFG['test_mean'] = mean
CFG['test_std'] = std

df_train = compute_stat_of_img(
    '../dataset/competition_dataset/facial_dataset/train', mode='train')
print('train dataset std : {:.4f} mean: {:.4f}'.format(
    (df_train['std'] / (df_train.R_max - df_train.R_min)).mean(),
    (df_train.R_avg / (df_train.R_max - df_train.R_min)).mean()))
CFG['channel_mean'] = (df_train.R_avg /
                       (df_train.R_max - df_train.R_min)).mean()
CFG['channel_std'] = (df_train['std'] /
                      (df_train.R_max - df_train.R_min)).mean()
```


```python
# def rm_duplicate():
#     img_data = []
#     for i,sample in enumerate(train_data):
#         image = sample[0]
#         hash_value = get_hash(image)
#         extrema = image.getextrema()
#         meta = {
#         'image': i,
#         'dataset': 'train',
#         'hash':  hash_value,
#         'R_min': extrema[0][0],
#         'R_max': extrema[0][1],
#         'G_min': extrema[1][0],
#         'G_max': extrema[1][1],
#         'B_min': extrema[2][0],
#         'B_max': extrema[2][1],
#         'height': image.height,
#         'width': image.width,
#         'format': image.format,
#         'mode': image.mode
#         }
#         img_data.append(meta)
#     return pd.DataFrame(img_data)
# df_train = rm_duplicate().head(10)
```


```python
# #查看重复数据
# dup = df_train.groupby(by='hash')[['dataset']].count().reset_index()
# dup = dup[dup['dataset']>1] #选择重复的数据
# dup.head()
```


```python
train_clean = df_train[['image','label','label_name']]
train_clean.head(10)
```


```python
test_clean = df_test[['image']]
test_clean.head(10)
```

### 数据可视化分析


```python
# #导入画图的库
# import plotly.express as px
# import plotly.graph_objects as go
# import plotly.figure_factory as ff
# import matplotlib.pyplot as plt

# #绘制样本类别分布图
# fig = px.histogram(train_clean,'label',marginal='violin', hover_data=train_clean.columns)
# fig.update_layout(title_text = 'Distribution of Classes')
# # fig.show()

# fig
# 由图可以看出 存在样本不均匀的问题
```


```python
# # 绘制不同通道像素分布直方图
# fig = ff.create_distplot([df_train['R_avg'],df_test['R_avg']],group_labels=['R','B'],colors=['RED','BLUE'])
# fig.update_layout(showlegend=False, template='simple_white')
# fig.update_layout(title_text="Distribution of Channel Values")

# #给柱状图添加黑色外框
# fig.data[0].marker.line.color = 'rgb(0, 0, 0)'
# fig.data[0].marker.line.width = 0.5
# fig.data[1].marker.line.color = 'rgb(0, 0, 0)'
# fig.data[1].marker.line.width = 0.5

# # fig
#由图可以看出 训练集和测试集的像素分布相同
```

### 采用imblearn缓解样本不平衡问题


```python
#采用上采样 增加少数类样本的数量
from imblearn.over_sampling import RandomOverSampler #随机重复取样
from imblearn.over_sampling import SMOTE             #选取少数类样本插值采样
from imblearn.over_sampling import BorderlineSMOTE   #边界类样本采样
from imblearn.over_sampling import ADASYN            #自适应合成采样

train_imb_data = train_clean[['image']]
train_imb_label = train_clean[['label']]
```


```python
def get_img(imgsrc):
    im_bgr = cv2.imread(imgsrc)
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    return im_rgb
```


```python
X = [get_img(os.path.join(CFG['train_root'],name)) for name in [*train_imb_data.image.values]]
y = train_imb_label.label
```


```python
imb_X = np.asarray(X).reshape(len(X), -1)
imb_y = y.values
```

#### 上采样策略选择


```python
# #采样策略 auto
# sm = ADASYN(random_state=CFG['seed'],n_jobs=-1)
# X_res, y_res = sm.fit_resample(imb_X, imb_y)
```


```python
#采样策略 auto
from imblearn.over_sampling import BorderlineSMOTE
sm = BorderlineSMOTE(random_state=CFG['seed'],n_jobs=-1,sampling_strategy={1:500})
X_res, y_res = sm.fit_resample(imb_X, imb_y)
```

#### 下采样策略选择


```python
# from imblearn.under_sampling import NearMiss
# ncr = NearMiss(n_jobs=-1)
# X_res,y_res = ncr.fit_resample(X_res,y_res)
```


```python
X_res.shape,y_res.shape
```


```python
CFG['idx_to_class']
```


```python
pd_res_y = pd.DataFrame({'label':y_res})
# fig = px.histogram(pd_res_y,'label',marginal='violin', hover_data=pd_res_y.columns)
# fig.update_layout(title_text = 'Distribution of Classes')
# fig
```


```python
X_res_imgs = X_res.reshape(-1,48,48,3)
X_res_imgs.shape
```


```python
y_res.shape
```

### 验证StratifiedKFold


```python
# folds = StratifiedKFold(n_splits=CFG['fold_num']).split(
#     np.arange(train_clean.shape[0]), train_clean.label.values)

# train_idx, valid_idx = next(iter(folds))

# train_fold = train_clean[['image','label_name']].iloc[train_idx,:]

# valid_fold = train_clean[['image','label_name']].iloc[valid_idx,:]
```


```python
# #绘制样本类别分布图
# fig = px.histogram(valid_fold,'label_name',marginal='violin', hover_data=valid_fold.columns)
# fig.update_layout(title_text = 'Distribution of trian_fold')
# fig.show()
```


```python
# #导入画图的库
# import plotly.express as px
# import plotly.graph_objects as go
# import plotly.figure_factory as ff
# import matplotlib.pyplot as plt

# #绘制样本类别分布图
# fig = px.histogram(train_fold,'label_name',marginal='violin', hover_data=train_fold.columns)
# fig.update_layout(title_text = 'Distribution of trian_fold')
# fig.show()

# # fig
# # 由图可以看出 存在样本不均匀的问题
```

### Dataset & DataLoader


```python
# train_clean = df_train[['image', 'label', 'label_name']]
# train_clean.head(10)
```


```python
# test_clean = df_test[['image']]
# test_clean.head(10)
```

### 固定随机种子


```python
#固定种子
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
```

### Dataset


```python
#数据类
class MyDataset(Dataset):
    def __init__(self, df, X_res_imgs,transforms=None, output_label=True):
        """
        Args:
            df():标签的df
            X_res_imgs():图片片集合 
            transforms():图片转换
            output_label():是否输出label
        """

        super().__init__()
        self.df = df.copy()
        self.transforms = transforms

        self.output_label = output_label
        self.X_res_imgs  = X_res_imgs[df.index]
        if output_label == True:
            self.labels = self.df['label'].values
        
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        # get labels
        if self.output_label:
            target = self.labels[index]
        
#         img = get_img("{}/{}".format(self.data_root,
#                                      self.df.loc[index]['image']))
#         img = X_res_imgs[index]
        img = self.X_res_imgs[index]
        #         img = edge_and_cut(img)
        #         assert len(img.shape)==3
        if self.transforms:
            img = self.transforms(image=img)['image']

        if self.output_label == True:
            return img, target
        else:
            return img
```

### TestDataset


```python
class TestDataset(Dataset):
    def __init__(self, df, data_root, transforms=None, output_label=True):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        
        self.output_label = output_label
        
        if output_label == True:
            self.labels = self.df['label'].values
            
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        # get labels
        if self.output_label:
            target = self.labels[index]
          
        img  = get_img("{}/{}".format(self.data_root, self.df.loc[index]['image']))
        
#         img = edge_and_cut(img)
#         assert len(img.shape)==3
        if self.transforms:
            img = self.transforms(image=img)['image']
        
        if self.output_label == True:
            return img, target
        else:
            return img
```

### Transforms


```python
#设定transforms
def get_train_transforms():
    return Compose(
        [
            OneOf([
                CoarseDropout(p=0.5),
                GaussNoise(),
            ], p=0.5),
            SomeOf(
                [
                    #             Transpose(p=0.5),
                    HorizontalFlip(p=0.5),
                    #             VerticalFlip(p=0.5),
                    ShiftScaleRotate(p=0.5,rotate_limit = [-15,15]),
                    HueSaturationValue(hue_shift_limit=0.2,
                                       sat_shift_limit=0.2,
                                       val_shift_limit=0.2,
                                       p=0.5),
                    RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                             contrast_limit=(-0.1, 0.1),
                                             p=0.5),
                ],
                n=3,
                p=0.6),
            Resize(CFG['width'], CFG['height']),
            Normalize(mean=[CFG['channel_mean']] * 3,
                      std=[CFG['channel_std']] * 3,
                      max_pixel_value=255.0,
                      p=1.0),
#             Normalize(mean=[0.485, 0.456, 0.406],
#                       std=[0.229, 0.224, 0.225],
#                       max_pixel_value=255.0,
#                       p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0)


def get_valid_transforms():
    return Compose([
        Resize(CFG['width'], CFG['height']),
        Normalize(mean=[CFG['channel_mean']] * 3,
                  std=[CFG['channel_std']] * 3,
                  max_pixel_value=255.0,
                  p=1.0),
#         Normalize(mean=[0.485, 0.456, 0.406],
#                   std=[0.229, 0.224, 0.225],
#                   max_pixel_value=255.0,
#                   p=1.0),
        ToTensorV2(p=1.0),
    ],
                   p=1.)
```

### Train&Valid DataLoader


```python
def prepare_dataloader(df, trn_idx, val_idx, X_res_imgs):

    train_ = df.loc[trn_idx, :]
    valid_ = df.loc[val_idx, :]

    train_ds = MyDataset(train_,
                         X_res_imgs,
                         transforms=get_train_transforms(),
                         output_label=True)
    valid_ds = MyDataset(valid_,
                         X_res_imgs,
                         transforms=get_valid_transforms(),
                         output_label=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG['train_bs'],
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=CFG['num_workers'],
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=True,
    )
    return train_loader, val_loader
```

## 定义模型


```python
import timm
import os
os.environ['TORCH_HOME'] = '/home/sun/data/yxx_notebook/pretrainModelHub'
net = timm.create_model('efficientnet_b3',pretrained=True,num_classes=7)
```


```python
#下载预训练模型
import torch
import os

os.environ['TORCH_HOME'] = '/home/sun/data/yxx_notebook/pretrainModelHub'


# get list of models
# torch.hub.list('zhanghang1989/ResNeSt', force_reload=False)
def get_pretrain_net():
    new_net = nn.Sequential()
    net = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
    new_net.fc = net
    new_net.output = nn.Sequential(nn.Dropout(),
                                   nn.Linear(1000, CFG['class_nums']))
    return new_net

#加入一个dropout层

# from torchinfo import summary

# batch_size = 128
# summary(net,input_size=(batch_size,3,48,48))
```


```python
# # #下载预训练模型
# # import torch
# # import os
# # import timm 
# # os.environ['TORCH_HOME'] = '/home/sun/data/yxx_notebook/pretrainModelHub'

# # # get list of models
# # # torch.hub.list('zhanghang1989/ResNeSt', force_reload=False)
# # def get_pretrain_net():
# #     net = timm.create_model('cspdarknet53',pretrained=True, num_classes=CFG['class_nums'])
# #     return net

# net = get_pretrain_net()
# # # from torchinfo import summary

# batch_size = 256
# summary(net,input_size=(batch_size,3,48,48))
```


```python
# # # model,_ = get_net_loss()
# X = torch.rand(128,3,48,48).to('cuda')
# # net.fc = nn.Linear(2048,7)
# net.to('cuda')
# net.eval()
# for layer in net.children():
#     print(layer)
#     X = layer(X)
#     print(X.shape)
# del net
```


```python
# from torchinfo import summary

# batch_size = 128
# summary(net,input_size=(batch_size,3,48,48))
```


```python
# timm.list_models() 
```

## 训练过程

### 定义focal loss


```python
from torch.autograd import Variable
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num=7, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
```

### train_one_epoch


```python
def train_one_epoch(epoch,
                    model,
                    loss_fn,
                    optimizer,
                    train_loader,
                    device,
                    scheduler=None):
    model.train()

    running_loss = None

    #     pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    for step, (imgs, image_labels) in enumerate(train_loader):
        
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        with autocast():
            image_preds = model(imgs)

            loss = loss_fn(image_preds, image_labels)
            scaler.scale(loss).backward()

            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * .99 + loss.item() * .01

            if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1)
                                                         == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if ((step + 1) % CFG['verbose_step']
                    == 0) or ((step + 1) == len(train_loader)):
                description = f'epoch {epoch} loss: {running_loss:.4f}'


#                 pbar.set_description(description)

    if scheduler is not None:
        scheduler.step()

    return running_loss
```

### valid_one_epoch


```python
def valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None):
    model.eval()

    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []
    val_loss = None

    #     pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in enumerate(val_loader):
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        image_preds = model(imgs)
        image_preds_all += [
            torch.argmax(image_preds, 1).detach().cpu().numpy()
        ]
        image_targets_all += [image_labels.detach().cpu().numpy()]

        loss = loss_fn(image_preds, image_labels)

        loss_sum += loss.item() * image_labels.shape[0]
        sample_num += image_labels.shape[0]

        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1)
                                                       == len(val_loader)):
            val_loss = loss_sum / sample_num
            description = f'epoch {epoch} loss: {val_loss:.4f}'


#             pbar.set_description(description)

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)

    val_acc = (image_preds_all == image_targets_all).mean()
    #     print('validation multi-class accuracy = {:.4f}'.format(val_acc))

    if scheduler is not None:
        scheduler.step()

    return val_acc, val_loss
```

### 训练主函数


```python
#main loop
if __name__ == '__main__':
    start = time.time()
    seed_everything(CFG['seed'])
    folds = StratifiedKFold(n_splits=CFG['fold_num'],
                            shuffle=True,
                            random_state=CFG['seed']).split(
                                np.arange(pd_res_y.shape[0]),
                                pd_res_y.label.values)
    train_loss_all = []
    val_loss_all = []
    val_acc_all = []
    for fold, (trn_idx, val_idx) in enumerate(folds):
        start_time = time.time()
        # 测试两个fold
        #         if fold > 2:
        #             break

        print('Training with {}th fold started'.format(fold))

        print(len(trn_idx), len(val_idx))

        #         fig = px.histogram(pd_res_y.loc[trn_idx],'label',marginal='violin', hover_data=pd_res_y.loc[trn_idx].columns)
        #         fig.update_layout(title_text = 'Distribution of Classes')
        #         fig.show()
        #         break

        train_loader, val_loader = prepare_dataloader(pd_res_y, trn_idx,
                                                      val_idx, X_res_imgs)

        device = torch.device(CFG['device'])

        model = get_pretrain_net().to(device)
        scaler = GradScaler()
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=CFG['lr'],
                                      weight_decay=CFG['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=CFG['T_0'],
            T_mult=2,
            eta_min=CFG['min_lr'],
            last_epoch=-1)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=CFG['epochs']-1)

        #         loss_tr = nn.CrossEntropyLoss().to(device)
        #         loss_fn = nn.CrossEntropyLoss().to(device)
        #使用focal loss
        loss_tr = FocalLoss().to(device)
        loss_fn = FocalLoss().to(device)

        train_loss_temp = []
        val_loss_temp = []
        val_acc_temp = []
        legend = ['train loss', 'val_loss', 'val_acc']
        animator = d2l.Animator(xlabel='epoch',
                                xlim=[1, CFG['epochs']],
                                legend=legend)
        for epoch in range(CFG['epochs']):
            train_loss_temp = train_one_epoch(epoch,
                                              model,
                                              loss_tr,
                                              optimizer,
                                              train_loader,
                                              device,
                                              scheduler=scheduler)

            train_loss_all.append([fold, epoch, train_loss_temp])
            with torch.no_grad():
                val_acc_temp, val_loss_temp = valid_one_epoch(epoch,
                                                              model,
                                                              loss_fn,
                                                              val_loader,
                                                              device,
                                                              scheduler=None)
                print('val acc:',val_acc_temp)
                val_loss_all.append([fold, epoch, val_loss_temp])
                val_acc_all.append([fold, epoch, val_acc_temp])
            animator.add(epoch + 1,
                         (train_loss_temp, val_loss_temp, val_acc_temp))
            if epoch > CFG['epochs'] - 4:  #save last three models
                torch.save(
                    model.state_dict(),
                    'models/{}_fold_{}_{}'.format(CFG['model_arch'], fold,
                                                  epoch))

        del model, optimizer, train_loader, val_loader, scaler, scheduler
        torch.cuda.empty_cache()
    seconds = time.time() - start
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("training time cost: %d时%02d分%02d秒" % (h, m, s))
```


```python
train_loss_all_df = pd.DataFrame(np.array(train_loss_all),
                                 columns=['fold', 'epoch', 'train_loss'])
val_loss_all_df = pd.DataFrame(np.array(val_loss_all),
                               columns=['fold', 'epoch', 'val_loss']) 
val_acc_all_df = pd.DataFrame(np.array(val_acc_all),
                              columns=['fold', 'epoch', 'val_acc'])
```


```python
val_acc_all_df.val_acc.values.max()
```

## 推理过程


```python
CFG['weights'] = [0.8,1,0.9] # weight for out model
CFG['tta'] = 3 # set TTA times
CFG['used_epochs'] = [CFG['epochs']-3,CFG['epochs']-2,CFG['epochs']-1] # choose the model
```

### 推理过程图片增强


```python
def get_inference_transforms():
    return Compose(
        [
            Resize(CFG['width'], CFG['height']),
            #             Transpose(p=0.5),
#             HorizontalFlip(p=0.5),
            #             VerticalFlip(p=0.5),
            #HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            #RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[CFG['test_mean']] * 3,
                      std=[CFG['test_std']] * 3,
                      max_pixel_value=255.0,
                      p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.)
```

### 推理过程的一个epoch


```python
def inference_one_epoch(model, data_loader, device):
    model.eval()

    image_preds_all = []

    #     pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in enumerate(data_loader):
        imgs = imgs.to(device).float()

        image_preds = model(imgs)
        image_preds_all += [
            torch.softmax(image_preds, 1).detach().cpu().numpy()
        ]

    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all
```

### 推理主函数


```python
if __name__ == '__main__':

    seed_everything(CFG['seed'])

    folds = StratifiedKFold(n_splits=CFG['fold_num']).split(
        np.arange(pd_res_y.shape[0]), pd_res_y.label.values)
    tst_preds_all = []
    for fold, (trn_idx, val_idx) in enumerate(folds):

#         if fold > 2:
#             break

        print('Inference fold {} started'.format(fold))

        valid_ = pd_res_y.loc[val_idx,:]
        valid_ds = MyDataset(valid_,
                             X_res_imgs,
                             transforms=get_inference_transforms(),
                             output_label=False)

        test_ds = TestDataset(test_clean,
                            CFG['test_root'],
                            transforms=get_inference_transforms(),
                            output_label=False)

        val_loader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=CFG['valid_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=True,
        )

        tst_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=CFG['valid_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=True,
        )

        device = torch.device(CFG['device'])
        model = get_pretrain_net().to(device)

        val_preds = []
        tst_preds = []

        for i, epoch in enumerate(CFG['used_epochs']):
            model.load_state_dict(
                torch.load('models/{}_fold_{}_{}'.format(
                    CFG['model_arch'], fold, epoch)))

            with torch.no_grad():
                for _ in range(CFG['tta']):
                    val_preds += [
                        CFG['weights'][i] / sum(CFG['weights']) / CFG['tta'] *
                        inference_one_epoch(model, val_loader, device)
                    ]
                    tst_preds += [
                        CFG['weights'][i] / sum(CFG['weights']) / CFG['tta'] *
                        inference_one_epoch(model, tst_loader, device)
                    ]

        val_preds = np.mean(val_preds, axis=0)
        tst_preds = np.mean(tst_preds, axis=0)  #计算不同epoch预测的平均值
        tst_preds_all.append(tst_preds)

        print('fold {} validation loss = {:.5f}'.format(
            fold, log_loss(valid_.label.values, val_preds)))
        print('fold {} validation accuracy = {:.5f}'.format(
            fold, (valid_.label.values == np.argmax(val_preds,
                                                    axis=1)).mean()))
        del model
        torch.cuda.empty_cache()
```

### 保存结果并展示


```python
avg_tst = np.mean(tst_preds_all, axis=0)  #计算不同折预测的平均值
test_clean['label'] = np.argmax(avg_tst, axis=1)
test_clean['label'] = [CFG['idx_to_class'][id] for id in test_clean.label]
test_clean['name'] = test_clean['image']
test_clean = test_clean[['name', 'label']]
test_clean.head(10)
```


```python
test_clean.value_counts('label')
```


```python
test_clean.to_csv('submission/submission_{}.csv'.format(CFG['file_name']),
                  index=False)
print('推理完成，文件submission_{}.csv已保存！'.format(CFG['file_name']))
```


```python
fig, axes = plt.subplots(2, 5, subplot_kw={'xticks': [], 'yticks': []})
axes = axes.flatten()
for i in range(10):
    axes[i].imshow(get_img(os.path.join(CFG['test_root'], df_test.iloc[i][0])))
plt.tight_layout()
```
