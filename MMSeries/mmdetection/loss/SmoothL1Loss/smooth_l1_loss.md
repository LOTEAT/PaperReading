<!--
 * @Author: LOTEAT
 * @Date: 2024-09-11 21:37:08
-->
- 前置知识：PyTorch, SmoothL1Loss
- [代码链接](https://github.com/open-mmlab/mmdetection)
### 1. SmoothL1Loss
`SmoothL1Loss`是L1损失和L2损失的组合，它通常在误差较小时使用L2损失，而在误差较大时使用L1损失。这种损失函数可以减少训练过程中对离群值的敏感性，同时保持对小误差的平滑性。

$$
SmoothL1Loss(x) = \begin{cases} 
0.5 \cdot x^2 & \text{if } |x| < \delta \\
|x| - 0.5 \cdot \delta & \text{otherwise}
\end{cases}
$$
其中$x$是预测值和真实值之间的差异（即残差），$\delta$是一个超参数，通常称为平滑阈值。

#### 2. Code
`SmoothL1Loss`通过`smooth_l1_loss`函数计算损失。
```python
@MODELS.register_module()
class SmoothL1Loss(nn.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 beta: float = 1.0,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        # 平滑阈值
        self.beta = beta
        self.reduction = reduction
        # smooth l1 loss的权重
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox
```
在`smooth_l1_loss`函数中，`weighted_loss`是个装饰器，是用来对损失求和、平均的。
```python
@weighted_loss
def smooth_l1_loss(pred: Tensor, target: Tensor, beta: float = 1.0) -> Tensor:
    """Smooth L1 loss.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        Tensor: Calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    # 和公式中是一样的
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss
```