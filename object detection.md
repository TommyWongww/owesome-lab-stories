---
style: plain
---

Object detection
===========
* [Object detection](#Object-detection )

    * [初步介绍](#初步介绍)

        * [思路](#思路)

            * [1暴力解决：滑动窗口](#1暴力解决)

            * [2Region Proposal](#2Region)

        * [RCNN](#RCNN)

        * [Fast R-CNN](#FastR-CNN)

        * [Faster RCNN](#FasterRCNN)

        * [YOLO](#YOLOYouOnlyLookOnce)

    * [论文目录](#论文目录)


## 初步介绍
卷积（边缘检测等等）
![20170325211712248](https://github.com/TommyWongww/owesome-lab-stories/blob/master/object%20detection.resource/20170325211712248.gif)

* 特征提取
![20180502180257](https://github.com/TommyWongww/owesome-lab-stories/blob/master/object%20detection.resource/20180502180257.png)

**Pooling（压缩）**
![20170325211641810](https://github.com/TommyWongww/owesome-lab-stories/blob/master/object%20detection.resource/20170325211641810.gif)

**Padding**
![20170325220130828](https://github.com/TommyWongww/owesome-lab-stories/blob/master/object%20detection.resource/20170325220130828.gif)

**重叠度（IOU）:**
物体检测需要定位出物体的bounding box，就像下面的图片一样，我们不仅要定位出车辆的bounding box 我们还要识别出bounding box 里面的物体就是车辆。

![v2-0659a27df35fd2f62cd00127ca8d1a21_b](https://github.com/TommyWongww/owesome-lab-stories/blob/master/object%20detection.resource/v2-0659a27df35fd2f62cd00127ca8d1a21_b.jpg)

对于bounding box的定位精度，有一个很重要的概念： 因为我们算法不可能百分百跟人工标注的数据完全匹配，因此就存在一个定位精度评价公式：IOU。 它定义了两个bounding box的重叠度，如下图所示：


![v2-6fe13f10a9cb286f06aa1e3e2a2b29bc_b](https://github.com/TommyWongww/owesome-lab-stories/blob/master/object%20detection.resource/v2-6fe13f10a9cb286f06aa1e3e2a2b29bc_b.jpg)

![v2-e26ffc0835bc30dede8d82989ef9e178_b](https://github.com/TommyWongww/owesome-lab-stories/blob/master/object%20detection.resource/v2-e26ffc0835bc30dede8d82989ef9e178_b.jpg)

**非极大值抑制（NMS**

![v2-19c03377416e437a288e29bd27e97c14_b](https://github.com/TommyWongww/owesome-lab-stories/blob/master/object%20detection.resource/v2-19c03377416e437a288e29bd27e97c14_b.jpg)

(1)从最大概率矩形框F开始，分别判断A~E与F的重叠度IOU是否大于某个设定的阈值;

(2)假设B、D与F的重叠度超过阈值，那么就扔掉B、D；并标记第一个矩形框F，是我们保留下来的。

(3)从剩下的矩形框A、C、E中，选择概率最大的E，然后判断E与A、C的重叠度，重叠度大于一定的阈值，那么就扔掉；并标记E是我们保留下来的第二个矩形框。

就这样一直重复，找到所有被保留下来的矩形框。


上下采样
VGG16  被很多目标检测用作特征提取
* 16 weight layers
* 13 convs + 3 fcs
* 5 conv blocks


```python
INPUT: [224x224x3]        memory:  224*224*3=150K   weights: 0
CONV3-64: [224x224x64]  memory:  224*224*64=3.2M   weights: (3*3*3)*64 = 1,728
CONV3-64: [224x224x64]  memory:  224*224*64=3.2M   weights: (3*3*64)*64 = 36,864
POOL2: [112x112x64]  memory:  112*112*64=800K   weights: 0
CONV3-128: [112x112x128]  memory:  112*112*128=1.6M   weights: (3*3*64)*128 = 73,728
CONV3-128: [112x112x128]  memory:  112*112*128=1.6M   weights: (3*3*128)*128 = 147,456
POOL2: [56x56x128]  memory:  56*56*128=400K   weights: 0
CONV3-256: [56x56x256]  memory:  56*56*256=800K   weights: (3*3*128)*256 = 294,912
![20170325211641810]($res/20170325211641810.gif)

CONV3-256: [56x56x256]  memory:  56*56*256=800K   weights: (3*3*256)*256 = 589,824
CONV3-256: [56x56x256]  memory:  56*56*256=800K   weights: (3*3*256)*256 = 589,824
POOL2: [28x28x256]  memory:  28*28*256=200K   weights: 0
CONV3-512: [28x28x512]  memory:  28*28*512=400K   weights: (3*3*256)*512 = 1,179,648
CONV3-512: [28x28x512]  memory:  28*28*512=400K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [28x28x512]  memory:  28*28*512=400K   weights: (3*3*512)*512 = 2,359,296
POOL2: [14x14x512]  memory:  14*14*512=100K   weights: 0
CONV3-512: [14x14x512]  memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [14x14x512]  memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [14x14x512]  memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
POOL2: [7x7x512]  memory:  7*7*512=25K  weights: 0
FC: [1x1x4096]  memory:  4096  weights: 7*7*512*4096 = 102,760,448
FC: [1x1x4096]  memory:  4096  weights: 4096*4096 = 16,777,216
FC: [1x1x1000]  memory:  1000 weights: 4096*1000 = 4,096,000
`````````
![20180502180245](https://github.com/TommyWongww/owesome-lab-stories/blob/master/object%20detection.resource/20180502180245.png)


## Object detection

dataset: PASCAL VOC 、ImageNet
PASCAL VOC 图片信息相对imageNet更加复杂

### 思路：

### 1.暴力解决：滑动窗口
（已经猜到人脸大小设定窗口大小）
![Lab-object-01](https://github.com/TommyWongww/owesome-lab-stories/blob/master/object%20detection.resource/Lab-object-01.png)
穷举来实现，每个像素遍历

### 2.Region Proposal（区域提名）


选择性搜索（颜色、纹理、尺度、包含关系）
color
texture
size
fill
进行合并
![Lab-object-02](https://github.com/TommyWongww/owesome-lab-stories/blob/master/object%20detection.resource/Lab-object-02.png)
光晕部分由色泽不同和图像之间的相似度，对其进行合并，最终分成几个大块儿，对应到下面就是这个蓝色的块儿，就是猜测到的有可能有目标物体的图块儿。然后对候选的几个框进行分类，这样极大提高了速度，从穷举变成有限次的分类问题。
类似层次聚类，本身是无监督的


一张图片 以提名的方式提出大概有目标物体的提名，然后用分类的方法去看看属于哪一类
SSD Driven Hierarchical clustering

rgb[^1]

### RCNN
用卷积神经网络CNN，在这里就是用Region-based CNN （RCNN）
![Lab-object-03](https://github.com/TommyWongww/owesome-lab-stories/blob/master/object%20detection.resource/Lab-object-03.png)

创新点：
* 采用CNN网络提取图像特征，从经验驱动的人造特征范式HOG、SIFT到数据驱动的表示学习范式，提高特征   对样本的表示能力；
* 采用大样本下有监督预训练+小样本微调的方式解决小样本难以训练甚至过拟合等问题。

**问题：**
* 近10年以来，以人工经验特征为主导的物体检测任务mAP【物体类别和位置的平均精度】提升缓慢
* 随着ReLu激励函数、dropout正则化手段和大规模图像样本集ILSVRC的出现，在2012年ImageNet大规模视觉识别挑战赛中，Hinton及他的学生采用CNN特征获得了最高的图像识别精确度；
* 上述比赛后，引发了一股“是否可以采用CNN特征来提高当前一直停滞不前的物体检测准确率“的热潮。


**流程:**
1.输入图像
2.每张图像生成1K~2K个候选区域
3.对每个候选区域，使用深度网络提取特征（AlexNet、VGG、CNN......）
4.1将特征送入每一类的SVM分类器，判别是否属于该类
4.2使用回归器精细修正候选框位置

pretrained with RCNN（用VGG16在imageNet）
嫁接到Pascal上，不用输出1000类，输出21类即可（加一个不在考虑的类别）
拿出卷积部分，再对最后的全连接层进行稍微改造即可



**候选框搜索阶段：**

当我们输入一张图片时，我们要搜索出所有可能是物体的区域，这里采用的就是前面提到的Selective Search方法，通过这个算法我们搜索出2000个候选框。然后从上面的总流程图中可以看到，搜出的候选框是矩形的，而且是大小各不相同。然而CNN对输入图片的大小是有固定的，如果把搜索到的矩形选框不做处理，就扔进CNN中，肯定不行。因此对于每个输入的候选框都需要缩放到固定的大小。下面我们讲解要怎么进行缩放处理，为了简单起见我们假设下一阶段CNN所需要的输入图片大小是个正方形图片227*227。因为我们经过selective search 得到的是矩形框，paper试验了两种不同的处理方法：

(1)各向异性缩放
(2)各向同性缩放
![v2-59449e8409b943f384c4cc3bf789d8b9_b](https://github.com/TommyWongww/owesome-lab-stories/blob/master/object%20detection.resource/v2-59449e8409b943f384c4cc3bf789d8b9_b.jpg)

A、先扩充后裁剪： 直接在原始图片中，把bounding box的边界进行扩展延伸成正方形，然后再进行裁剪；如果已经延伸到了原始图片的外边界，那么就用bounding box中的颜色均值填充；如上图(B)所示;

B、先裁剪后扩充：先把bounding box图片裁剪出来，然后用固定的背景颜色填充成正方形图片(背景颜色也是采用bounding box的像素颜色均值),如上图(C)所示;

对于上面的异性、同性缩放，文献还有个padding处理，上面的示意图中第1、3行就是结合了padding=0,第2、4行结果图采用padding=16的结果。经过最后的试验，作者发现采用各向异性缩放、padding=16的精度最高。

**a、网络结构设计阶段**

网络架构两个可选方案：第一选择经典的Alexnet；第二选择VGG16。经过测试Alexnet精度为58.5%，VGG16精度为66%。VGG这个模型的特点是选择比较小的卷积核、选择较小的跨步，这个网络的精度高，不过计算量是Alexnet的7倍。后面为了简单起见，我们就直接选用Alexnet，并进行讲解；Alexnet特征提取部分包含了5个卷积层、2个全连接层，在Alexnet中p5层神经元个数为9216、 f6、f7的神经元个数都是4096，通过这个网络训练完毕后，最后提取特征每个输入候选框图片都能得到一个4096维的特征向量。

**b、网络有监督预训练阶段 （图片数据库：ImageNet ILSVC ）**

fine-tune（采用AlexNet）优化采用随机梯度下降法，学习率：0.001



CNN阶段：打标签，selective search VS 人工
				选取IOU>0.5 + or - 样本


 
**c、SVM训练、测试阶段**

**训练阶段：**

这是一个二分类问题，我么假设我们要检测车辆。我们知道只有当bounding box把整量车都包含在内，那才叫正样本；如果bounding box 没有包含到车辆，那么我们就可以把它当做负样本。但问题是当我们的检测窗口只有部分包含物体，那该怎么定义正负样本呢？
作者测试了IOU阈值各种方案数值0,0.1,0.2,0.3,0.4,0.5。最后通过训练发现，如果选择IOU阈值为0.3效果最好（选择为0精度下降了4个百分点，选择0.5精度下降了5个百分点）,即当重叠度小于0.3的时候，我们就把它标注为负样本。一旦CNN f7层特征被提取出来，那么我们将为每个物体类训练一个svm分类器。当我们用CNN提取2000个候选框，可以得到2000*4096这样的特征向量矩阵，然后我们只需要把这样的一个矩阵与svm权值矩阵4096*N点乘(N为分类类别数目，因为我们训练的N个svm，每个svm包含了4096个权值w)，就可以得到结果了。



![v2-3ef21dd028fd210f92107c1ded528045_b](https://github.com/TommyWongww/owesome-lab-stories/blob/master/object%20detection.resource/v2-3ef21dd028fd210f92107c1ded528045_b.jpg)


**位置精修：** 
目标检测问题的衡量标准是重叠面积：许多看似准确的检测结果，往往因为候选框不够准确，重叠面积很小。故需要一个位置精修步骤。 
回归器：对每一类目标，使用一个线性脊回归器进行精修。正则项λ=10000。 输入为深度网络pool5层的4096维特征，输出为xy方向的缩放和平移。 
训练样本：判定为本类的候选框中和真值重叠面积大于0.6的候选框。




对一张图要提名的太多2000，然后对每一个提名做卷积提取特征，也就是2000次，那么运行效率会比较慢
引出Fast R-CNN。



### Fast R-CNN
一个图片经过卷积后的特征映射，信息量降低了，但对物体的分类的重要信息都保留下来了，所以只对特征映射做区域提名，进一步的卷积提取特征，不在原图上做区域提名，因此减少了计算量。

selected search是在cpu上运行的，因此想到对区域提名部分再做一个卷积操作，使整个操作都能在GPU上完成。

### Faster RCNN


两个分支：区域提名；特征提取以及方框位置的分类的计算。（框的位置，框内部是什么）
一眼：看框打在哪儿。
两眼：看框内是什么

为了考虑实时性，达到在线预测的目的。

### YOLO（You Only Look Once）
整张图像输入，直接输出回归层 Bounding Box的位置和Bounding Box所属的类别

将一幅图像分成SxS个网格(grid cell)，如果某个object的中心落在这个网格中，则这个网格就负责预测这个object。

缺点：当被检测目标集中在某个区域时效果很差。因为每个格子只能对应预测一个bounding box和类别概率，因此无法预测邻近区域的物体，检测精度下降了。

SSD（Single Shot Multi-box Detector）
predict object（物体），以及其 归属类别的 score（得分）；同时在 feature map 上使用小的卷积核，去 predict 一系列 bounding boxes 的 box offsets。

目标检测再进一步------行为动作捕捉。








## 论文目录：

[2013-----RCNN----Rich feature hierarchies for accurate object detection and semantic segmentation](https://zhuanlan.zhihu.com/p/29936564)

2013----Deep Neural Networks for Object Detection

2014----SPPnet----Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition-sppnet

[2015----Fast R-CNN](https://zhuanlan.zhihu.com/p/29953111)

[2016----Faster R-CNN-Towards Real-Time Object Detection with Region Proposal Networks](https://zhuanlan.zhihu.com/p/29969145)

2016----Inside-Outside Net_Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks

[2016----R-FCN-Object Detection via Region-based Fully Convolutional Networks](https://zhuanlan.zhihu.com/p/30788068)

[2016----SSD-Single Shot MultiBox Detector](https://zhuanlan.zhihu.com/p/30478644)

[目标检测之SSD代码分析（MXNet版）](https://zhuanlan.zhihu.com/p/30553929)

2016----YOLO9000-better,faster,stronger

2016----You Only Look Once-Unified, Real-Time Object Detection

2017----A-Fast-RCNN_Hard Positive Generation via Adversary for Object Detection

[2017----Deformable Convolutional Networks](https://zhuanlan.zhihu.com/p/30927896)

2017----DSOD_ Learning Deeply Supervised Object Detectors from Scratch

[2017----Focal Loss for Dense Object Detection](https://zhuanlan.zhihu.com/p/30701067)

2017----Mask R-CNN

2017----Speed_Accuracy trade-offs for modern convolutional object detectors

[2017----Light-Head R-CNN_In Defense of Two-Stage Object Detector](https://zhuanlan.zhihu.com/p/31389174)



[^1]: http://www.rossgirshick.info/

