<!--
 * @Author: LOTEAT
 * @Date: 2024-07-23 17:11:31
-->
# PaperReading
在人工智能的学习中，我们需要掌握大量的神经网络架构。在学习这些架构的时候，我经常会看的时候掌握，但是没几天就忘记了。所以，我创建了这个仓库。我将我自己在论文学习以及代码学习的过程中掌握的知识以要点的形式写出来。在整理过程中，我发现有些论文可以被划归为很多个类别，因此在不同类别中可能会出现重复的论文讲解。

持续更新，与诸君共勉。

## 主要内容
### 1. Computer Vision


#### 1. Backbone

##### 1.1 Basic Knowlege
- [ ] <a href='ComputerVision/Basic/Transformer/transformer.md'>Transformer</a>: 等待更新


##### 2.1 Backbone Networks
- [x] <a href='ComputerVision/Backbone/ResNet/resnet.md'>ResNet</a>: 残差网络，其残差思想已经被广泛应用当今的网络设计中。

- [x] <a href='ComputerVision/Backbone/MobileNet/mobilenet.md'>MobileNet</a>: 采用`Depthwise Separable Convolution`降低计算量。

- [x] <a href='ComputerVision/Backbone/MobileNetV2/mobilenetv2.md'>MobileNetV2</a>: 采用linear layer替换ReLU，并引入残差连接，在降低参数计算量的同时提高性能。

- [x] <a href='ComputerVision/Backbone/MobileNetV3/mobilenetv3.md'>MobileNetV3</a>: 引入通道注意力机制，并采用新的激活函数，在降低参数计算量的同时提高性能。

- [x] <a href='ComputerVision/Backbone/EfficientNet/efficientnet.md'>EfficientNet</a>: 探究了深度、宽度以及分辨率对性能的影响。


#### 2. 2D Detection

##### 2.1 Basic Knowlege

- [ ] <a href='ComputerVision/Basic/mmdection2d.md'>MMDetection2d</a>: MMDetection2D是由MMLab编写的一款2D目标检测框架，在本文档中将简略介绍一些会用到的关键基础通用组件。

#### 3. 3D Detection

##### 3.1 Basic Knowlege

- [ ] <a href='ComputerVision/Basic/mmdection3d.md'>MMDetection3d</a>: MMDetection3D是由MMLab编写的一款3D目标检测框架，在本文档中将简略介绍一些会用到的关键基础通用组件。

##### 3.2 BEV

- [ ] <a href='ComputerVision/3DObjectDetection/BEV/Basic/lss.md'>Lift, Splat, and Shoot</a>: 等待更新


### 2. Natural Language Processing

### 3. Math

- [x] <a href='Math/Jacobian/jacobian.md'>Jacobian Matrix</a>: Jacobian矩阵推导

- [x] <a href='Math/MultivariateGaussianDensity/multivariate_gaussian_density.md'>Multivariate Gaussian Density</a>: 多元高斯函数密度函数推导

- [x] <a href='Math/GaussianAddition/gaussian_addition.md'>Addition of Gaussian Distribution</a>: 高斯分布相加后概率密度函数推导

### 4. Localization
- [x] <a href='Localization/BayesFilter/bayes_filter.md'>Bayes Filter</a>: Bayes Filter推导

- [x] <a href='Localization/KalmanFilter/kalman_filter.md'>Kalman Filter</a>: Kalman Filter推导

- [x] <a href='Localization/ExtendedKalmanFilter/extended_kalman_filter.md'>Extended Kalman Filter</a>: Extended Kalman Filter推导


### 5. Motion Planning
#### 5.1 Search Based Methods
- [x] <a href='MotionPlanning/SearchBased/AStar/astar.md'>AStar</a>: 启发式搜索算法。

- [x] <a href='MotionPlanning/SearchBased/BFS/bfs.md'>BFS</a>: 宽度优先搜索。

- [x] <a href='MotionPlanning/SearchBased/DFS/dfs.md'>DFS</a>: 广度优先搜索。

#### 5.1 Sampling Based Methods
- [x] <a href=''>RRT</a>: 基于随机采样的路径搜索算法。