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

- [x] <a href='ComputerVision/2DObjectDetection/Basic/IoU/iou.md'>IoU</a>: IoU计算
- [ ] <a href='ComputerVision/2DObjectDetection/Basic/mAP/mAP.md'>mAP</a>: mAP计算
- [ ] <a href='ComputerVision/2DObjectDetection/Basic/NMS/nms.md'>NMS</a>: 非极大值抑制

##### 2.2 Two Stage

- [x] <a href='ComputerVision/2DObjectDetection/RCNN/rcnn.md'>RCNN</a>: RPN网络老祖，但是还没有完全提出RPN

- [x] <a href='ComputerVision/2DObjectDetection/FastRCNN/fast_rcnn.md'>Fast RCNN</a>: 对RCNN网络进行优化

- [ ] <a href='ComputerVision/2DObjectDetection/FasterRCNN/faster_rcnn.md'>Faster RCNN</a>: RPN网络开山之作

- [ ] <a href='ComputerVision/2DObjectDetection/MaskRCNN/mask_rcnn.md'>Mask RCNN</a>: 

- [x] <a href='ComputerVision/2DObjectDetection/FeaturePyramidNetwork/feature_pyramid_network.md'>FeaturePyramidNetwork</a>: 融合多尺度特征，为不同尺度的物体提供丰富的语义信息


##### 2.3 One Stage



#### 3. 3D Detection


##### 3.1 Basic Knowlege

- [ ] <a href='ComputerVision/Basic/mmdection3d.md'>MMDetection3d</a>: MMDetection3D是由MMLab编写的一款3D目标检测框架，在本文档中将简略介绍一些会用到的关键基础通用组件。

##### 3.2 BEV

###### 3.2.1 Basic Knowlege

- [x] <a href='ComputerVision/3DObjectDetection/BEV/Basic/LSS/lss.md'>Lift, Splat, and Shoot</a>: 通过预测图片像素的深度信息，构建了2D到3D的空间转换，将2D图像转化为BEV图像。

###### 3.2.2 Camera-based Method
- [ ] <a href='ComputerVision/3DObjectDetection/BEV/BEVDepth/bevdepth.md'>BEVDepth</a>: 等待更新

- [ ] <a href='ComputerVision/3DObjectDetection/BEV/BEVFormer/bevformer.md'>BEVFormer</a>: 等待更新

- [ ] <a href='ComputerVision/3DObjectDetection/BEV/BEVFusion/bevfusion.md'>BEVFusion</a>: 等待更新

###### 3.2.3 LiDAR-based Method

###### 3.2.4 Fusion-based Method

- [ ] <a href='ComputerVision/3DObjectDetection/BEV/BEVFusion/bevfusion.md'>BEVFusion</a>: 等待更新


#### 4. Occupancy Network

##### 4.1 Basic Knowlege
- [x] <a href='ComputerVision/Occupancy/Basic/SemanticKITTI/semantic_kitti.md'>SemanticKITTI</a>: SemanticKITTI数据格式

- [ ] <a href='ComputerVision/Occupancy/Basic/ContextPrior/context_prior.md'>ContextPrior</a>: 



##### 4.2 Occupancy Network
- [x] <a href='ComputerVision/Occupancy/Basic/ContextPrior/context_prior.md'>CPNet</a>: 亲和力损失。

- [x] <a href='ComputerVision/Occupancy/OccupancyNetwork/occupancy_network.md'>Occupancy Network</a>: 将占据栅格网络引入到3维重建中，为之后的世界模型打下了基础。

- [x] <a href='ComputerVision/Occupancy/MonoScene/mono_scene.md'>MonoScene Network</a>: 单目相机的Occupancy Network

- [ ] <a href='ComputerVision/Occupancy/SurroundOcc/surround_occ.md'>SurroundOcc Network</a>: 





### 2. Natural Language Processing

### 3. Math

- [x] <a href='Math/Jacobian/jacobian.md'>Jacobian Matrix</a>: Jacobian矩阵推导

- [x] <a href='Math/MultivariateGaussianDensity/multivariate_gaussian_density.md'>Multivariate Gaussian Density</a>: 多元高斯函数密度函数推导

- [x] <a href='Math/GaussianAddition/gaussian_addition.md'>Addition of Gaussian Distribution</a>: 高斯分布相加后概率密度函数推导

- [x] <a href='Math/DubinsCurve/dubins_curve.md'>Dubins Curve</a>: 杜宾斯曲线

-  <a href='https://github.com/LOTEAT/DubinsPath-Demo'> Dubins Path Demo</a>: 杜宾斯曲线Demo

- [ ] <a href='Math/BézierCurve/bézier_curve.md'>Bézier Curve</a>: 贝塞尔曲线

### 4. Localization
- [x] <a href='Localization/BayesFilter/bayes_filter.md'>Bayes Filter</a>: Bayes Filter推导

- [x] <a href='Localization/KalmanFilter/kalman_filter.md'>Kalman Filter</a>: Kalman Filter推导

- [x] <a href='Localization/ExtendedKalmanFilter/extended_kalman_filter.md'>Extended Kalman Filter</a>: Extended Kalman Filter推导

-  <a href='https://github.com/LOTEAT/EKF-Demo'> Extended Kalman Filter Demo</a>: 扩展卡尔曼滤波Demo


### 5. Motion Planning

#### 5.1 Basic Knowlege
- [x] <a href='MotionPlanning/Basic/OBVP/obvp.md'> OBVP</a>: OBVP问题及求解

- [x] <a href='MotionPlanning/Basic/PathPlanning/path_planning.md'> PathPlanning</a>: PathPlanning公共组件



#### 5.2 Search Based Methods
- [x] <a href='MotionPlanning/SearchBased/AStar/astar.md'>AStar</a>: 启发式搜索算法。

- [x] <a href='MotionPlanning/SearchBased/BFS/bfs.md'>BFS</a>: 宽度优先搜索。

- [x] <a href='MotionPlanning/SearchBased/DFS/dfs.md'>DFS</a>: 广度优先搜索。

#### 5.3 Sampling Based Methods
- [x] <a href='MotionPlanning/SamplingBased/RRT/rrt.md'>RRT</a>: 基于随机采样的路径搜索算法。

#### 5.4 Motion Model
- [x] <a href='MotionPlanning/MotionModel/OdometryMotionModel/odometry_motion_model.md'>Odometry Motion Model</a>: 里程计模型
- <a href='https://github.com/LOTEAT/OdometryModelDemo'> Odometry Motion Model Demo</a>: 里程计模型Demo

- [x] <a href='MotionPlanning/MotionModel/VelocityMotionModel/velocity_motion_model.md'>Velocity Motion Model</a>: 速度模型

#### 5.5 Trajectory Generation

- [x] <a href='MotionPlanning/TrajectoryGeneration/MinimumSnap/minimum_snap.md'>Minimum Snap</a>: minimum snap轨迹生成

### 6. MMSeries
#### 6.1 MMEngine

##### 6.1.1 Scheduler
- [x] <a href='MMSeries/mmengine/scheduler/Basic/scheduler.md'>Scheduler</a>: MMEngine Scheduler基类

- [x] <a href='MMSeries/mmengine/scheduler/LinearScheduler/linear_scheduler.md'>Linear Scheduler</a>: 线性学习率调整

- [x] <a href='MMSeries/mmengine/scheduler/LinearScheduler/linear_scheduler.md'>StepLR Scheduler</a>: 步长学习率调整

- [x] <a href='MMSeries/mmengine/scheduler/MultiStepScheduler/multi_step_scheduler.md'>Multi Step Scheduler</a>: 多步长学习率调整

#### 6.2 MMDetection

##### 6.2.1 Task Utils
- [x] <a href='MMSeries/mmdetection/task_utils/Basic/Assigner/assigner.md'>Assigner</a>: MMDetection Assigner基类

- [x] <a href='MMSeries/mmdetection/task_utils/Basic/AssignResult/assign_result.md'>AssignResult</a>: 包装Assign后的结果

- [x] <a href='MMSeries/mmdetection/task_utils/MaxIoUAssigner/max_iou_assigner.md'>MaxIouAssigner</a>: 基于IoU的标签赋值

- [x] <a href='MMSeries/mmdetection/task_utils/Basic/Sampler/sampler.md'>Sampler</a>: MMDetection Sampler基类

- [x] <a href='MMSeries/mmdetection/task_utils/Basic/SamplingResult/sampling_result.md'>SamplerResult</a>: 包装Sample后的结果


#### 6.3 MMDetection3D