<!--
 * @Author: LOTEAT
 * @Date: 2024-10-15 19:23:16
-->
## You Only Look Once: Unified, Real-Time Object Detection
- 前置知识：PyTorch
- 作者：Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
- [文章链接](https://arxiv.org/pdf/1506.02640)
- [代码链接](https://github.com/yjh0410/PyTorch_YOLOv1.git)

### 1. Motivation
在先前的目标检测算法中，很多都是两阶段的检测算法。也就是先使用RPN网络提出region proposal，然后再进行检测。而Yolov1则尝试使用端到端的方式生成检测框。

### 2. Architecture
