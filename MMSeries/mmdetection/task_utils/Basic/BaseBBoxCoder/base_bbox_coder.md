<!--
 * @Author: LOTEAT
 * @Date: 2024-09-06 15:00:12
-->
## BaseBBoxCoder 
- 前置知识：PyTorch
- [代码链接](https://github.com/open-mmlab/mmdetection)
### 1. Code
`BBoxCoder`的作用就是自定义回归框的编解码机制。这是因为不同算法最后的输出解码可能是不一样的，我们需要不同的编解码函数。
```python
class BaseBBoxCoder(metaclass=ABCMeta):
    """Base bounding box coder.

    Args:
        use_box_type (bool): Whether to warp decoded boxes with the
            box type data structure. Defaults to False.
    """

    # The size of the last of dimension of the encoded tensor.
    encode_size = 4

    def __init__(self, use_box_type: bool = False, **kwargs):
        self.use_box_type = use_box_type

    @abstractmethod
    def encode(self, bboxes, gt_bboxes):
        """Encode deltas between bboxes and ground truth boxes."""

    @abstractmethod
    def decode(self, bboxes, bboxes_pred):
        """Decode the predicted bboxes according to prediction and base
        boxes."""

```