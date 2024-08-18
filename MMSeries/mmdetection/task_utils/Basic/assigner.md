<!--
 * @Author: LOTEAT
 * @Date: 2024-08-17 16:50:59
-->
## Scheduler 
- 前置知识：PyTorch
- [代码链接](https://github.com/open-mmlab/mmdetection)
在`mmdetection`中`Assigner`类的主要作用是使用不同的策略，为bbox赋予正标签（类别标签）或者是负标签（背景标签）。可以看到，基类的代码十分简单，只有一个assign函数供后续子类调用。
```python
class BaseAssigner(metaclass=ABCMeta):
    """Base assigner that assigns boxes to ground truth boxes."""

    @abstractmethod
    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs):
        """Assign boxes to either a ground truth boxes or a negative boxes."""
```