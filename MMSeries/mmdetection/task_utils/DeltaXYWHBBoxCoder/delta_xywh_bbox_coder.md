<!--
 * @Author: LOTEAT
 * @Date: 2024-09-06 15:07:38
-->
## DeltaXYWHBBoxCoder 
- 前置知识：PyTorch, <a href='../Basic/BaseBBoxCoder/base_bbox_coder.md'> BaseBBoxCoder </a>, <a href='../../../../ComputerVision/2DObjectDetection/RCNN/rcnn.md'> R-CNN </a>
- [代码链接](https://github.com/open-mmlab/mmdetection)
### 1. DeltaXYWH And BBox
注意，我们在R-CNN中提到，最后预测的偏移公式是：
$$
\begin{aligned}
& \hat{G}_x=P_w d_x(P)+P_x \\
& \hat{G}_y=P_h d_y(P)+P_y \\
& \hat{G}_w=P_w \exp \left(d_w(P)\right) \\
& \hat{G}_h=P_h \exp \left(d_h(P)\right) .
\end{aligned}
$$
这个类也是基于这个公式实现的。

### 2. Code
讲解在代码注释中。

#### 2.1 DeltaXYWHBBoxCoder
```python
@TASK_UTILS.register_module()
class DeltaXYWHBBoxCoder(BaseBBoxCoder):
    """Delta XYWH BBox coder.

    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    this coder encodes bbox (x1, y1, x2, y2) into delta (dx, dy, dw, dh) and
    decodes delta (dx, dy, dw, dh) back to original bbox (x1, y1, x2, y2).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
        add_ctr_clamp (bool): Whether to add center clamp, when added, the
            predicted box is clamped is its center is too far away from
            the original anchor's center. Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
            Default 32.
    """

    def __init__(self,
                 target_means: Sequence[float] = (0., 0., 0., 0.),
                 target_stds: Sequence[float] = (1., 1., 1., 1.),
                 clip_border: bool = True,
                 add_ctr_clamp: bool = False,
                 ctr_clamp: int = 32,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.means = target_means
        self.stds = target_stds
        self.clip_border = clip_border
        self.add_ctr_clamp = add_ctr_clamp
        self.ctr_clamp = ctr_clamp

    def encode(self, bboxes: Union[Tensor, BaseBoxes],
               gt_bboxes: Union[Tensor, BaseBoxes]) -> Tensor:
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor or :obj:`BaseBoxes`): Source boxes,
                e.g., object proposals.
            gt_bboxes (torch.Tensor or :obj:`BaseBoxes`): Target of the
                transformation, e.g., ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """
        # 获得bbox的tensor
        bboxes = get_box_tensor(bboxes)
        # 获得gt的tensor，实际上这个gt是预测值
        gt_bboxes = get_box_tensor(gt_bboxes)
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
        encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(
        self,
        bboxes: Union[Tensor, BaseBoxes],
        pred_bboxes: Tensor,
        max_shape: Optional[Union[Sequence[int], Tensor,
                                  Sequence[Sequence[int]]]] = None,
        wh_ratio_clip: Optional[float] = 16 / 1000
    ) -> Union[Tensor, BaseBoxes]:
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            bboxes (torch.Tensor or :obj:`BaseBoxes`): Basic boxes. Shape
                (B, N, 4) or (N, 4)
            pred_bboxes (Tensor): Encoded offsets with respect to each roi.
               Has shape (B, N, num_classes * 4) or (B, N, 4) or
               (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H
               when rois is a grid of anchors.Offset encoding follows [1]_.
            max_shape (Sequence[int] or torch.Tensor or Sequence[
               Sequence[int]],optional): Maximum bounds for boxes, specifies
               (H, W, C) or (H, W). If bboxes shape is (B, N, 4), then
               the max_shape should be a Sequence[Sequence[int]]
               and the length of max_shape should also be B.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            Union[torch.Tensor, :obj:`BaseBoxes`]: Decoded boxes.
        """
        # 获得bboxes的tensor
        bboxes = get_box_tensor(bboxes)
        assert pred_bboxes.size(0) == bboxes.size(0)
        if pred_bboxes.ndim == 3:
            assert pred_bboxes.size(1) == bboxes.size(1)

        if pred_bboxes.ndim == 2 and not torch.onnx.is_in_onnx_export():
            # single image decode
            # 参考下面的delta2bbox注释
            decoded_bboxes = delta2bbox(bboxes, pred_bboxes, self.means,
                                        self.stds, max_shape, wh_ratio_clip,
                                        self.clip_border, self.add_ctr_clamp,
                                        self.ctr_clamp)
        else:
            # 不太了解onnx，等学了再看
            # 虽然学了之后大概率也懒得再补充了 >_<
            if pred_bboxes.ndim == 3 and not torch.onnx.is_in_onnx_export():
                warnings.warn(
                    'DeprecationWarning: onnx_delta2bbox is deprecated '
                    'in the case of batch decoding and non-ONNX, '
                    'please use “delta2bbox” instead. In order to improve '
                    'the decoding speed, the batch function will no '
                    'longer be supported. ')
            decoded_bboxes = onnx_delta2bbox(bboxes, pred_bboxes, self.means,
                                             self.stds, max_shape,
                                             wh_ratio_clip, self.clip_border,
                                             self.add_ctr_clamp,
                                             self.ctr_clamp)

        if self.use_box_type:
            assert decoded_bboxes.size(-1) == 4, \
                ('Cannot warp decoded boxes with box type when decoded boxes'
                 'have shape of (N, num_classes * 4)')
            decoded_bboxes = HorizontalBoxes(decoded_bboxes)
        return decoded_bboxes
```

#### 2.2 bbox2delta
这个函数是实际上就是delta2bbox的反函数。
```python
def bbox2delta(
    proposals: Tensor,
    gt: Tensor,
    means: Sequence[float] = (0., 0., 0., 0.),
    stds: Sequence[float] = (1., 1., 1., 1.)
) -> Tensor:
    """Compute deltas of proposals w.r.t. gt.

    We usually compute the deltas of x, y, w, h of proposals w.r.t ground
    truth bboxes to get regression target.
    This is the inverse function of :func:`delta2bbox`.

    Args:
        proposals (Tensor): Boxes to be transformed, shape (N, ..., 4)
        gt (Tensor): Gt bboxes to be used as base, shape (N, ..., 4)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates

    Returns:
        Tensor: deltas with shape (N, 4), where columns represent dx, dy,
            dw, dh.
    """
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0]
    gh = gt[..., 3] - gt[..., 1]

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas
```

#### 2.3 delta2bbox
```python
def delta2bbox(rois: Tensor,
               deltas: Tensor,
               means: Sequence[float] = (0., 0., 0., 0.),
               stds: Sequence[float] = (1., 1., 1., 1.),
               max_shape: Optional[Union[Sequence[int], Tensor,
                                         Sequence[Sequence[int]]]] = None,
               wh_ratio_clip: float = 16 / 1000,
               clip_border: bool = True,
               add_ctr_clamp: bool = False,
               ctr_clamp: int = 32) -> Tensor:
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 4).
        deltas (Tensor): Encoded offsets relative to each roi.
            Has shape (N, num_classes * 4) or (N, 4). Note
            N = num_base_anchors * W * H, when rois is a grid of
            anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates.
            Default (0., 0., 0., 0.).
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates. Default (1., 1., 1., 1.).
        max_shape (tuple[int, int]): Maximum bounds for boxes, specifies
           (H, W). Default None.
        wh_ratio_clip (float): Maximum aspect ratio for boxes. Default
            16 / 1000.
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Default True.
        add_ctr_clamp (bool): Whether to add center clamp. When set to True,
            the center of the prediction bounding box will be clamped to
            avoid being too far away from the center of the anchor.
            Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
            Default 32.

    Returns:
        Tensor: Boxes with shape (N, num_classes * 4) or (N, 4), where 4
           represent tl_x, tl_y, br_x, br_y.

    References:
        .. [1] https://arxiv.org/abs/1311.2524

    Example:
        >>> rois = torch.Tensor([[ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 5.,  5.,  5.,  5.]])
        >>> deltas = torch.Tensor([[  0.,   0.,   0.,   0.],
        >>>                        [  1.,   1.,   1.,   1.],
        >>>                        [  0.,   0.,   2.,  -1.],
        >>>                        [ 0.7, -1.9, -0.5,  0.3]])
        >>> delta2bbox(rois, deltas, max_shape=(32, 32, 3))
        tensor([[0.0000, 0.0000, 1.0000, 1.0000],
                [0.1409, 0.1409, 2.8591, 2.8591],
                [0.0000, 0.3161, 4.1945, 0.6839],
                [5.0000, 5.0000, 5.0000, 5.0000]])
    """
    # 注意，deltas的维度是(N, 4*num_classes)
    num_bboxes, num_classes = deltas.size(0), deltas.size(1) // 4
    if num_bboxes == 0:
        return deltas

    deltas = deltas.reshape(-1, 4)
    # 这个means和stds我也不知道有什么实际意义，可能是trick
    # 如果mean设置为0， stds设置为1，和原始R-CNN是一样的
    means = deltas.new_tensor(means).view(1, -1)
    stds = deltas.new_tensor(stds).view(1, -1)
    denorm_deltas = deltas * stds + means

    dxy = denorm_deltas[:, :2]
    dwh = denorm_deltas[:, 2:]

    # Compute width/height of each roi
    rois_ = rois.repeat(1, num_classes).reshape(-1, 4)
    # 获得中心点坐标
    pxy = ((rois_[:, :2] + rois_[:, 2:]) * 0.5)
    # 获得w和h
    pwh = (rois_[:, 2:] - rois_[:, :2])
    # 加入xy的偏移
    dxy_wh = pwh * dxy
    # add_ctr_clamp是为了防止偏移量过大的情况，所以加以限制
    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dxy_wh = torch.clamp(dxy_wh, max=ctr_clamp, min=-ctr_clamp)
        dwh = torch.clamp(dwh, max=max_ratio)
    else:
        dwh = dwh.clamp(min=-max_ratio, max=max_ratio)
    # 中心坐标转化
    gxy = pxy + dxy_wh
    # 加入wh的偏移量
    gwh = pwh * dwh.exp()
    # 获得左上角和右下角的坐标
    x1y1 = gxy - (gwh * 0.5)
    x2y2 = gxy + (gwh * 0.5)
    bboxes = torch.cat([x1y1, x2y2], dim=-1)
    # 超出边界框进行裁剪
    if clip_border and max_shape is not None:
        bboxes[..., 0::2].clamp_(min=0, max=max_shape[1])
        bboxes[..., 1::2].clamp_(min=0, max=max_shape[0])
    bboxes = bboxes.reshape(num_bboxes, -1)
    return bboxes
```