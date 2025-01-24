<!--
 * @Author: LOTEAT
 * @Date: 2025-01-24 10:25:36
-->
# Yolov2学习笔记

## You Only Look Once: Improved Real-Time Object Detection
- 前置知识：PyTorch, <a href='./../FasterRCNN/faster_rcnn.md'>Faster RCNN</a>
- 作者：Joseph Redmon, Ali Farhadi
- [文章链接](https://arxiv.org/pdf/1612.08242)
- [代码链接](https://github.com/AlexeyAB/darknet)

### 1. Motivation
YOLOv2是YOLOv1的改进版本，旨在提高检测精度和速度。通过引入新的网络结构和训练策略，YOLOv2在保持实时检测速度的同时，显著提高了检测精度。YOLOv2还引入了YOLO9000模型，可以检测超过9000多类物体。

YOLOv1算法的缺点包括：
- 在物体定位方面不够准确
- 难以找到图片中的所有物体，召回率较低
- 检测小目标和密集目标性能较差
- 虽然速度快，但mAP准确度较低

### 2. Architecture
#### 2.1 Batch Normalization
在YOLOv2中，每个卷积层后面添加了Batch Normalization层，不再使用dropout。实验证明添加了BN层可以提高2%的mAP。

#### 2.2 High Resolution
YOLOv2在ImageNet数据集上使用高分辨率输入进行finetune，提高了模型在检测数据集上的适应性，mAP提升了约4%。也就是从$224\times 224变成了448 \times 448$。

#### 2.3 Anchor
YOLOv2引入了锚框机制，借鉴了Faster R-CNN的思想，使用不同大小和宽高比的边框来覆盖图像的不同位置和多种尺度。

#### 2.4 Cluster
YOLOv2对训练集中标注的边框进行K-mean聚类分析，以寻找匹配样本的边框尺寸。

#### 2.5 直接位置预测
YOLOv2使用sigmoid函数处理偏移值，将边界框中心点约束在当前cell中，防止偏移过多。

#### 2.6 细粒度特征
YOLOv2通过添加passthrough layer，将前一层的26x26特征图与当前层的13x13特征图连接，提高了小目标的检测能力。

#### 2.7 多尺度训练
YOLOv2引入了多尺度训练策略，使模型能够适应不同大小的输入图像。

#### 2.8 更快的网络
YOLOv2采用了新的基础模型Darknet-19，包括19个卷积层和5个maxpooling层，使用了1x1卷积来压缩特征图channels以降低计算量和参数。
