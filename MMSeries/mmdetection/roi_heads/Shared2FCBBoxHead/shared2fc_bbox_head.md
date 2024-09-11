<!--
 * @Author: LOTEAT
 * @Date: 2024-09-11 23:05:32
-->
## Shared2FCBBoxHead
- 前置知识：PyTorch，<a href="../ConvFCBBoxHead/conv_fc_bbox_head.md">ConvFCBBoxHead</a>
- [代码链接](https://github.com/open-mmlab/mmdetection)

### 1. Code
`Shared2FCBBoxHead`是`ConvFCBBoxHead`一种实现。
```python
@MODELS.register_module()
class Shared2FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels: int = 1024, *args, **kwargs) -> None:
        super().__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
```