# Yolov3学习笔记

## YOLOv3: An Incremental Improvement
- 前置知识：PyTorch, <a href="./../Yolov2/yolov2.md">YOLOv2</a>
- 作者：Joseph Redmon, Ali Farhadi
- [文章链接](https://arxiv.org/pdf/1804.02767.pdf)
- [代码链接](https://github.com/ultralytics/yolov3)

### 1. Motivation
YOLOv3是YOLOv2的改进版本，主要目标是在保持快速检测速度的同时进一步提高检测精度。相比YOLOv2，YOLOv3在以下几个方面进行了改进：
- 使用更强大的特征提取网络
- 采用多尺度预测
- 改进了分类损失函数
- 提供了不同规模的模型以适应不同场景

### 2. Architecture
#### 2.1 Darknet-53
YOLOv3采用了全新的backbone网络Darknet-53，包含53个卷积层，使用了残差连接。相比Darknet-19，新的网络结构更深，特征提取能力更强。主要特点：
- 使用连续的3×3和1×1卷积层
- 添加残差连接（类似ResNet）
- 使用步长为2的卷积层进行下采样
- 每个卷积层后都使用BN层和Leaky ReLU激活函数
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="yolov3.assets/darknet53.png" width = "100%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      图1：Darknet-53
  	</div>
</center>



#### 2.2 多尺度预测
YOLOv3在三个不同尺度上进行预测：
- 大尺度特征图(52×52)：检测小目标
- 中等尺度特征图(26×26)：检测中等大小目标
- 小尺度特征图(13×13)：检测大目标

每个尺度上使用3个不同的anchor box，总共使用9个anchor box。

#### 2.3 特征金字塔网络
YOLOv3借鉴了FPN的思想，采用了类似的特征金字塔结构：
- 使用上采样和特征融合
- 高层特征语义信息强，低层特征位置信息准
- 不同层级特征的融合提高了检测效果

#### 2.4 预测过程改进
每个边界框预测：
- 4个边界框坐标(tx, ty, tw, th)
- 1个objectness score
- 80个类别置信度（使用sigmoid代替softmax）

#### 2.5 损失函数
分类损失使用二元交叉熵损失代替了多类别交叉熵损失：
- 支持多标签分类
- 每个类别独立预测概率
- 不同类别之间不互斥

#### 2.6 数据流
整个数据流可以表征为（原图见[知乎](https://zhuanlan.zhihu.com/p/76802514)）：
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="yolov3.assets/dataflow.png" width = "100%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      图2：dataflow
  	</div>
</center>

