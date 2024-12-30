<!--
 * @Author: LOTEAT
 * @Date: 2024-10-14 15:32:30
-->
## NMS
- PyTorch, CUDA
- [代码链接](https://github.com/LOTEAT/mmcv-ops/blob/main/mmcv_ops/nms/nms.ipynb)

### 1. NMS
在目标检测当中，常用的一种后处理手段就是NMS，也就是非极大值抑制。有很多种非极大值抑制的方法，这里基于mmcv中的代码进行学习。

NMS，其步骤可以概括如下：
- 设定好IoU阈值
- 对bbox按照置信度排序
- 双重循环
- - 第一重循环先对排序后bbox进行遍历，如果当前的bbox已经被舍弃则跳过
- - 第二重循环，从第一重循环遍历的bbox下一个开始遍历
- - - 如果当前的bbox已经被舍弃则跳过
- - - 如果两个bbox的IoU超过阈值，则舍弃第二个bbox
- 结束循环，保留结果

### 2. 

### 3. Code
参考[代码](https://github.com/LOTEAT/mmcv-ops/blob/main/mmcv_ops/nms/nms.ipynb)。