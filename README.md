# [信也科技杯图像算法大赛——智能零售柜商品识别](https://ai.ppdai.com/mirror/goToMirrorDetailSix?mirrorId=26&tabindex=1)

## 赛题描述及数据说明 

<font face="楷体" size=4>本次大赛的数据为静态智能零售货柜采集摆放商品后的零售柜内部图片，在此基础上进行人工标注给出了该次竞赛的数据集，数据集分为**训练数据集**、**训练商品库**、**初赛评估集**、**初赛商品库**、**复赛评估集**和**复赛商品库**，初赛及复赛中均可使用训练数据集、训练商品库。   


本竞赛的**目标**是检测出商品的外接矩形框且识别出商品的类别（通过从商品库中检索出最相似的商品信息来确定待识别的商品类别）。  

因训练数据集、初赛评估集及复赛评估集中包含的图片都包含多个商品实例，该类图片称为密集商品图片；商品库中包含的任一图片只包含一个商品实例，该类图片称为稀疏商品图片。  


初赛环节提供训练数据集、训练商品库、初赛评估集、初赛商品库供选手下载。  


训练数据集包含图片数据及标注信息（标注信息详见“数据集下载”页面的数据集说明）。图片数据集为密集商品图片，格式为jpg。标注信息遵循COCO数据集的标注格式。  


训练商品库包含了图片数据集及标注信息。图片数据集为稀疏商品图片，格式为jpg，每张图片仅包含一个商品，同一商品在商品库中有多张图片，尽量覆盖各商品在柜中不同位置及角度。标注信息遵循COCO数据集的标注格式。训练时，可依据训练数据中的商品实例类别id与商品库中商品实例类别id进行关联。  

**具体详情请看比赛官网！**

<font face="华文彩云" size=4>害，他们比赛拿钱，而我只想躺着借数据van.....

## 数据处理

### 解压数据集


```python
!unzip -oq /home/aistudio/data/data91732/train.zip -d dataset
```

### 处理数据集

<font face="楷体" size=4>虽然官方说标注信息遵循COCO数据集的标注格式，不过我用COCO API读取的时候还是有点问题，再加上最近想用PaddleX 客户端试试，所以就转成了VOC格式。　　

这里有一个[格式转换及数据处理的仓库正在建设](https://github.com/thomas-yanxin/Scripts-about-datasets)，各位有兴趣的可以看看，欢迎PR！！！！　　

以下是COCO转VOC的具体操作：


```python
!git clone https://gitee.com/yanxin_thomas/cocoapi.git
```


```python
!pip install pycocotools -i https://mirror.baidu.com/pypi/simple
```


```python
%cd cocoapi/PythonAPI/
!make
!python setup.py install
!pip install lxml
```


```python
%cd dataset/
!mkdir VOC
```


```python
%cd /home/aistudio/
```


```python
from pycocotools.coco import COCO #这个包可以从git上下载https://github.com/cocodataset/cocoapi/tree/master/PythonAPI，也可以直接用修改后的coco.py
import  os, cv2, shutil
from lxml import etree, objectify
from tqdm import tqdm
from PIL import Image
 
CKimg_dir ='/home/aistudio/dataset/VOC/JPEGImages'
CKanno_dir = '/home/aistudio/dataset/VOC/Annotations'

#若模型保存文件夹不存在，创建模型保存文件夹，若存在，删除重建
def mkr(path):
        os.mkdir(path)
 
def save_annotations(filename, objs,filepath):
    annopath = CKanno_dir + "/" + filename[:-3] + "xml" #生成的xml文件保存路径
    dst_path = CKimg_dir + "/" + filename
    img_path=filepath
    img = cv2.imread(img_path)
    im = Image.open(img_path)
    if im.mode != "RGB":
        print(filename + " not a RGB image")
        im.close()
        return
    im.close()
    shutil.copy(img_path, dst_path)#把原始图像复制到目标文件夹
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('1'),
        E.filename(filename),
        E.source(
            E.database('CKdemo'),
            E.annotation('VOC'),
            E.image('CK')
        ),
        E.size(
            E.width(img.shape[1]),
            E.height(img.shape[0]),
            E.depth(img.shape[2])
        ),
        E.segmented(0)
    )
    for obj in objs:
        E2 = objectify.ElementMaker(annotate=False)
        anno_tree2 = E2.object(
            E.name(obj[0]),
            E.pose(),
            E.truncated("0"),
            E.difficult(0),
            E.bndbox(
                E.xmin(obj[2]),
                E.ymin(obj[3]),
                E.xmax(obj[4]),
                E.ymax(obj[5])
            )
        )
        anno_tree.append(anno_tree2)
    etree.ElementTree(anno_tree).write(annopath, pretty_print=True)
 
 
def showbycv(coco, dataType, img, classes,origin_image_dir,verbose=False):
    filename = img['file_name']
    filepath=os.path.join(origin_image_dir,dataType,filename)
    I = cv2.imread(filepath)
    annIds = coco.getAnnIds(imgIds=img['id'],  iscrowd=None)
    anns = coco.loadAnns(annIds)
    objs = []
    for ann in anns:
        name = classes[ann['category_id']]
        if 'bbox' in ann:
            bbox = ann['bbox']
            xmin = (int)(bbox[0])
            ymin = (int)(bbox[1])
            xmax = (int)(bbox[2] + bbox[0])
            ymax = (int)(bbox[3] + bbox[1])
            obj = [name, 1.0, xmin, ymin, xmax, ymax]
            objs.append(obj)
            if verbose:
                cv2.rectangle(I, (xmin, ymin), (xmax, ymax), (255, 0, 0))
                cv2.putText(I, name, (xmin, ymin), 3, 1, (0, 0, 255))
    save_annotations(filename, objs,filepath)
    if verbose:
        cv2.imshow("img", I)
        cv2.waitKey(0)
 
 
def catid2name(coco):#将名字和id号建立一个字典
    classes = dict()
    for cat in coco.dataset['categories']:
        classes[cat['id']] = cat['name']
        # print(str(cat['id'])+":"+cat['name'])
    return classes
 
 
def get_CK5(origin_anno_dir,origin_image_dir,verbose=False):
    dataTypes = ['a_images']
    for dataType in dataTypes:
        annFile = 'a_annotations.json'
        annpath=os.path.join(origin_anno_dir,annFile)
        coco = COCO(annpath)
        classes = catid2name(coco)
        imgIds = coco.getImgIds()
        # imgIds=imgIds[0:1000]#测试用，抽取10张图片，看下存储效果
        for imgId in tqdm(imgIds):
            img = coco.loadImgs(imgId)[0]
            showbycv(coco, dataType, img, classes,origin_image_dir,verbose=False)
 
def main():
    base_dir='/home/aistudio/dataset/VOC' #step1 这里是一个新的文件夹，存放转换后的图片和标注
    image_dir=os.path.join(base_dir,'JPEGImages')#在上述文件夹中生成images，annotations两个子文件夹
    anno_dir=os.path.join(base_dir,'Annotations')
    mkr(image_dir)
    mkr(anno_dir)
    origin_image_dir='/home/aistudio/dataset'#step 2原始的coco的图像存放位置
    origin_anno_dir='/home/aistudio/dataset'#step 3 原始的coco的标注存放位置
    verbose=True #是否需要看下标记是否正确的开关标记，若是true,就会把标记展示到图片上
    get_CK5(origin_anno_dir,origin_image_dir,verbose)
 
if __name__ == "__main__":
    main()
```


```python

from pycocotools.coco import COCO #这个包可以从git上下载https://github.com/cocodataset/cocoapi/tree/master/PythonAPI，也可以直接用修改后的coco.py
import  os, cv2, shutil
from lxml import etree, objectify
from tqdm import tqdm
from PIL import Image
 
CKimg_dir ='/home/aistudio/dataset/VOC/JPEGImages'
CKanno_dir = '/home/aistudio/dataset/VOC/Annotations'


def save_annotations(filename, objs,filepath):
    annopath = CKanno_dir + "/" + filename[:-3] + "xml" #生成的xml文件保存路径
    dst_path = CKimg_dir + "/" + filename
    img_path=filepath
    img = cv2.imread(img_path)
    im = Image.open(img_path)
    if im.mode != "RGB":
        print(filename + " not a RGB image")
        im.close()
        return
    im.close()
    shutil.copy(img_path, dst_path)#把原始图像复制到目标文件夹
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('1'),
        E.filename(filename),
        E.source(
            E.database('CKdemo'),
            E.annotation('VOC'),
            E.image('CK')
        ),
        E.size(
            E.width(img.shape[1]),
            E.height(img.shape[0]),
            E.depth(img.shape[2])
        ),
        E.segmented(0)
    )
    for obj in objs:
        E2 = objectify.ElementMaker(annotate=False)
        anno_tree2 = E2.object(
            E.name(obj[0]),
            E.pose(),
            E.truncated("0"),
            E.difficult(0),
            E.bndbox(
                E.xmin(obj[2]),
                E.ymin(obj[3]),
                E.xmax(obj[4]),
                E.ymax(obj[5])
            )
        )
        anno_tree.append(anno_tree2)
    etree.ElementTree(anno_tree).write(annopath, pretty_print=True)
 
 
def showbycv(coco, dataType, img, classes,origin_image_dir,verbose=False):
    filename = img['file_name']
    filepath=os.path.join(origin_image_dir,dataType,filename)
    I = cv2.imread(filepath)
    annIds = coco.getAnnIds(imgIds=img['id'],  iscrowd=None)
    anns = coco.loadAnns(annIds)
    objs = []
    for ann in anns:
        name = classes[ann['category_id']]
        if 'bbox' in ann:
            bbox = ann['bbox']
            xmin = (int)(bbox[0])
            ymin = (int)(bbox[1])
            xmax = (int)(bbox[2] + bbox[0])
            ymax = (int)(bbox[3] + bbox[1])
            obj = [name, 1.0, xmin, ymin, xmax, ymax]
            objs.append(obj)
            if verbose:
                cv2.rectangle(I, (xmin, ymin), (xmax, ymax), (255, 0, 0))
                cv2.putText(I, name, (xmin, ymin), 3, 1, (0, 0, 255))
    save_annotations(filename, objs,filepath)
    if verbose:
        cv2.imshow("img", I)
        cv2.waitKey(0)
 
 
def catid2name(coco):#将名字和id号建立一个字典
    classes = dict()
    for cat in coco.dataset['categories']:
        classes[cat['id']] = cat['name']
        # print(str(cat['id'])+":"+cat['name'])
    return classes
 
 
def get_CK5(origin_anno_dir,origin_image_dir,verbose=False):
    dataTypes = ['b_images']
    for dataType in dataTypes:
        annFile = 'b_annotations.json'
        annpath=os.path.join(origin_anno_dir,annFile)
        coco = COCO(annpath)
        classes = catid2name(coco)
        imgIds = coco.getImgIds()
        # imgIds=imgIds[0:1000]#测试用，抽取10张图片，看下存储效果
        for imgId in tqdm(imgIds):
            img = coco.loadImgs(imgId)[0]
            showbycv(coco, dataType, img, classes,origin_image_dir,verbose=False)
 
def main():
    base_dir='/home/aistudio/dataset/VOC' #step1 这里是一个新的文件夹，存放转换后的图片和标注
    image_dir=os.path.join(base_dir,'JPEGImages')#在上述文件夹中生成images，annotations两个子文件夹
    anno_dir=os.path.join(base_dir,'Annotations')
    origin_image_dir='/home/aistudio/dataset'#step 2原始的coco的图像存放位置
    origin_anno_dir='/home/aistudio/dataset'#step 3 原始的coco的标注存放位置
    verbose=True #是否需要看下标记是否正确的开关标记，若是true,就会把标记展示到图片上
    get_CK5(origin_anno_dir,origin_image_dir,verbose)
 
if __name__ == "__main__":
    main()
```

这里检查是否有图像的命名中不符合规范的（例如有空格），有就删掉。别问我为啥特指空格，当然是因为踩过坑……


```python
import os 
path_1 = 'dataset/VOC/JPEGImages'
path_2 = 'dataset/VOC/Annotations'
file_lists = os.listdir(path_1)
# print(file_lists)
for i in file_lists:
    if " " in i:
        new = i.replace(' ','')
        print(new)
        os.rename(os.path.join(path,i),os.path.join(path,new))
        # print(new)
file_lists = os.listdir(path_2)
# print(file_lists)
for i in file_lists:
    if " " in i:
        new = i.replace(' ','')
        print(new)
        os.rename(os.path.join(path,i),os.path.join(path,new))
        # print(new)
```

<font face="楷体" size=4>安装PaddleX的依赖包


```python
!pip install paddlex
!pip install paddle2onnx
```

以７：２：１化分训练集、验证集、测试集


```python
!paddlex --split_dataset --format VOC --dataset_dir /home/aistudio/dataset/VOC/ --val_value 0.2 --test_value 0.1
```

## 训练辽！！！　　

<font face="楷体" size=4>在使用PaddleX进行模型训练的过程中，我们使用目前PaddleX适配精度最高的PPYolo模型进行训练。其模型较大，预测速度比YOLOv3-DarkNet53更快，适用于服务端。大家也可以更改其他模型尝试一下。训练了420轮，最终map可以达到80％以上，还算可以趴……（自我安慰中）




```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from paddlex.det import transforms
import paddlex as pdx

# 定义训练和验证时的transforms
# API说明 https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html
train_transforms = transforms.Compose([
    transforms.MixupImage(mixup_epoch=250), transforms.RandomDistort(),
    transforms.RandomExpand(), transforms.RandomCrop(), transforms.Resize(
        target_size=608, interp='RANDOM'), transforms.RandomHorizontalFlip(),
    transforms.Normalize()
])

eval_transforms = transforms.Compose([
    transforms.Resize(
        target_size=608, interp='CUBIC'), transforms.Normalize()
])

# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-vocdetection
train_dataset = pdx.datasets.VOCDetection(
    data_dir='/home/aistudio/dataset/VOC',
    file_list='/home/aistudio/dataset/VOC/train_list.txt',
    label_list='/home/aistudio/dataset/VOC/labels.txt',
    transforms=train_transforms,
    parallel_method='thread',
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='/home/aistudio/dataset/VOC',
    file_list='/home/aistudio/dataset/VOC/val_list.txt',
    label_list='/home/aistudio/dataset/VOC/labels.txt',
    parallel_method='thread',
    transforms=eval_transforms)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://paddlex.readthedocs.io/zh_CN/develop/train/visualdl.html
num_classes = len(train_dataset.labels)

# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-ppyolo
model = pdx.det.PPYOLO(num_classes=num_classes)

# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#train
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
model.train(
    num_epochs=540,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    learning_rate=0.000125,
    save_interval_epochs=1,
    lr_decay_epochs=[270,320, 480],
    save_dir='output/ppyolo',
    use_vdl=True)
```

![](https://ai-studio-static-online.cdn.bcebos.com/22bd3b70da164af09b59fcf50168918e721dd7f8273e46ada3576fd529fb5144)


## 模型导出　

<font face="楷体" size=4>这里我们将训练过程中保存的模型导出为inference格式模型，其原因在于：PaddlePaddle框架保存的权重文件分为两种：支持前向推理和反向梯度的训练模型 和 只支持前向推理的推理模型。二者的区别是推理模型针对推理速度和显存做了优化，裁剪了一些只在训练过程中才需要的tensor，降低显存占用，并进行了一些类似层融合，kernel选择的速度优化。而导出的inference格式模型包括__model__、__params__和model.yml三个文件，分别表示模型的网络结构、模型权重和模型的配置文件（包括数据预处理参数等）。


```python
!paddlex --export_inference --model_dir=output/ppyolo/best_model --save_dir=./inference_model
```

## 批量预测


```python
test_list = []
with open("/home/aistudio/dataset/VOC/test_list.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')  #去掉列表中每一个元素的换行符
        line = line.split(' ')[0]
        # print(line)
        test_list.append('dataset/VOC/' + line)
print(test_list)
    
```


```python
import paddlex as pdx
import os
predictor = pdx.deploy.Predictor('/home/aistudio/inference_model')

# # print(img_list)
num_list = len(test_list)
list_1 = test_list[:int(num_list/4)]
print(list_1)
# print(len(list_1))
result = predictor.batch_predict(image_list=list_1)
for i in result:
    print(i)
# print(result)
```

## 后期规划

* 　　用PaddleDetection试试，那个选择多；　　

* 	等我有板子了还是想部署在板子上做个东西出来……　　
* 　有时间的话还是按比赛的要求试试趴，毕竟钱也不少……
