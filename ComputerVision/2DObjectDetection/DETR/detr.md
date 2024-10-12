<!--
 * @Author: LOTEAT
 * @Date: 2024-10-10 11:13:08
-->

## End-to-End Object Detection with Transformers
- 前置知识：<a href="../../Backbone/Basic/Transformer/transformer.md">Transformer</a>
- 作者：Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko
- [文章链接](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460205.pdf)
- [代码链接](https://github.com/open-mmlab/mmdetection)
- [视频链接](https://www.bilibili.com/video/BV1GB4y1X72R/?spm_id_from=333.337.search-card.all.click&vd_source=962a14fe78f4c5b36a73df66a4d2f23b)

### 1. Introduction
这篇论文bryanyzhu老师已经讲解得非常仔细了，推荐去看原视频。

### 2. Code
我们先来看配置文件。
```python
model = dict(
    type='DETR',
    num_queries=100,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=None,
        num_outs=1),
    encoder=dict(  # DetrTransformerEncoder
        num_layers=6,
        layer_cfg=dict(  # DetrTransformerEncoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)))),
    decoder=dict(  # DetrTransformerDecoder
        num_layers=6,
        layer_cfg=dict(  # DetrTransformerDecoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            cross_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True))),
        return_intermediate=True),
    positional_encoding=dict(num_feats=128, normalize=True),
    bbox_head=dict(
        type='DETRHead',
        num_classes=80,
        embed_dims=256,
        loss_cls=dict(
            type='CrossEntropyLoss',
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='ClassificationCost', weight=1.),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=100))
```
模型是`DETR`。按照`DetectionTransformer`的架构，模型应该如下面所示：
```shell
                 img_feats & batch_data_samples
                               |
                               V
                      +-----------------+
                      | pre_transformer |
                      +-----------------+
                          |          |
                          |          V
                          |    +-----------------+
                          |    | forward_encoder |
                          |    +-----------------+
                          |             |
                          |             V
                          |     +---------------+
                          |     |  pre_decoder  |
                          |     +---------------+
                          |         |       |
                          V         V       |
                      +-----------------+   |
                      | forward_decoder |   |
                      +-----------------+   |
                                |           |
                                V           V
                               head_inputs_dict
```
代码及注释如下：
```python
@MODELS.register_module()
class DETR(DetectionTransformer):
    r"""Implementation of `DETR: End-to-End Object Detection with Transformers.

    <https://arxiv.org/pdf/2005.12872>`_.

    Code is modified from the `official github repo
    <https://github.com/facebookresearch/detr>`_.
    """

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        # 针对图像特征的embedding
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DetrTransformerEncoder(**self.encoder)
        self.decoder = DetrTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        # NOTE The embed_dims is typically passed from the inside out.
        # For example in DETR, The embed_dims is passed as
        # self_attn -> the first encoder layer -> encoder -> detector.
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def pre_transformer(
            self,
            img_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        """Prepare the inputs of the Transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            img_feats (Tuple[Tensor]): Tuple of features output from the neck,
                has shape (bs, c, h, w).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such as
                `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict, dict]: The first dict contains the inputs of encoder
            and the second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              and 'feat_pos'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask',
              and 'memory_pos'.
        """

        feat = img_feats[-1]  # NOTE img_feats contains only one feature.
        batch_size, feat_dim, _, _ = feat.shape
        # construct binary masks which for the transformer.
        assert batch_data_samples is not None
        # 检查每个图像的形状 
        # 如果批次中的图像形状不同，需要构建一个掩码张量来表示无效位置
        batch_input_shape = batch_data_samples[0].batch_input_shape
        input_img_h, input_img_w = batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        same_shape_flag = all([
            s[0] == input_img_h and s[1] == input_img_w for s in img_shape_list
        ])
        if torch.onnx.is_in_onnx_export() or same_shape_flag:
            masks = None
            # [batch_size, embed_dim, h, w]
            pos_embed = self.positional_encoding(masks, input=feat)
        else:
            masks = feat.new_ones((batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                img_h, img_w = img_shape_list[img_id]
                masks[img_id, :img_h, :img_w] = 0
            # NOTE following the official DETR repo, non-zero values represent
            # ignored positions, while zero values mean valid positions.
            # 掩码插值，因为特征的分辨率一般会经过图像的下采样
            masks = F.interpolate(
                masks.unsqueeze(1),
                size=feat.shape[-2:]).to(torch.bool).squeeze(1)
            # [batch_size, embed_dim, h, w]
            pos_embed = self.positional_encoding(masks)

        # use `view` instead of `flatten` for dynamically exporting to ONNX
        # [bs, c, h, w] -> [bs, h*w, c]
        feat = feat.view(batch_size, feat_dim, -1).permute(0, 2, 1)
        pos_embed = pos_embed.view(batch_size, feat_dim, -1).permute(0, 2, 1)
        # [bs, h, w] -> [bs, h*w]
        if masks is not None:
            masks = masks.view(batch_size, -1)

        # prepare transformer_inputs_dict
        # 将特征、掩码和位置嵌入放入字典中，分别返回给编码器和解码器
        encoder_inputs_dict = dict(
            feat=feat, feat_mask=masks, feat_pos=pos_embed)
        decoder_inputs_dict = dict(memory_mask=masks, memory_pos=pos_embed)
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor) -> Dict:
        """Forward with Transformer encoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            feat (Tensor): Sequential features, has shape (bs, num_feat_points,
                dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (bs, num_feat_points).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (bs, num_feat_points, dim).

        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output.
        """
        memory = self.encoder(
            query=feat, query_pos=feat_pos,
            key_padding_mask=feat_mask)  # for self_attn
        encoder_outputs_dict = dict(memory=memory)
        return encoder_outputs_dict

    def pre_decoder(self, memory: Tensor) -> Tuple[Dict, Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).

        Returns:
            tuple[dict, dict]: The first dict contains the inputs of decoder
            and the second dict contains the inputs of the bbox_head function.

            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'query', 'query_pos',
              'memory'.
            - head_inputs_dict (dict): The keyword args dictionary of the
              bbox_head functions, which is usually empty, or includes
              `enc_outputs_class` and `enc_outputs_class` when the detector
              support 'two stage' or 'query selection' strategies.
        """

        batch_size = memory.size(0)  # (bs, num_feat_points, dim)
        query_pos = self.query_embedding.weight
        # (num_queries, dim) -> (bs, num_queries, dim)
        query_pos = query_pos.unsqueeze(0).repeat(batch_size, 1, 1)
        query = torch.zeros_like(query_pos)

        decoder_inputs_dict = dict(
            query_pos=query_pos, query=query, memory=memory)
        head_inputs_dict = dict()
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self, query: Tensor, query_pos: Tensor, memory: Tensor,
                        memory_mask: Tensor, memory_pos: Tensor) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional queries of decoder inputs,
                has shape (bs, num_queries, dim).
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            memory_pos (Tensor): The positional embeddings of memory, has
                shape (bs, num_feat_points, dim).

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output.

            - hidden_states (Tensor): Has shape
              (num_decoder_layers, bs, num_queries, dim)
        """

        hidden_states = self.decoder(
            query=query,
            key=memory,
            value=memory,
            query_pos=query_pos,
            key_pos=memory_pos,
            key_padding_mask=memory_mask)  # for cross_attn

        head_inputs_dict = dict(hidden_states=hidden_states)
        return head_inputs_dict
```

Positional Encoding。
```python
@MODELS.register_module()
class SinePositionalEncoding(BaseModule):
    """Position encoding with sine and cosine functions.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self,
                 num_feats: int,
                 temperature: int = 10000,
                 normalize: bool = False,
                 scale: float = 2 * math.pi,
                 eps: float = 1e-6,
                 offset: float = 0.,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask: Tensor, input: Optional[Tensor] = None) -> Tensor:
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].
            input (Tensor, optional): Input image/feature Tensor.
                Shape [bs, c, h, w]

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        assert not (mask is None and input is None)

        if mask is not None:
            # 代码首先利用掩码来确定图像中的有效位置
            # 并计算这些位置在x和y方向的累积和
            # 从而实现像素的区分
            B, H, W = mask.size()
            device = mask.device
            # For convenience of exporting to ONNX,
            # it's required to convert
            # `masks` from bool to int.
            mask = mask.to(torch.int)
            not_mask = 1 - mask  # logical_not
            y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)
        else:
            # 在没有mask的情况下，会通过torch.arange生成坐标矩阵：
            # single image or batch image with no padding
            B, _, H, W = input.shape
            device = input.device
            x_embed = torch.arange(
                1, W + 1, dtype=torch.float32, device=device)
            x_embed = x_embed.view(1, 1, -1).repeat(B, H, 1)
            y_embed = torch.arange(
                1, H + 1, dtype=torch.float32, device=device)
            y_embed = y_embed.view(1, -1, 1).repeat(B, 1, W)
        if self.normalize:
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self) -> str:
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'eps={self.eps})'
        return repr_str
```
整体流程其实并不复杂，如果去看encoder和decoder layer，也是标准的transformer架构。

DETR另一个重点是使用匈牙利进行匹配。这里代码是使用scipy库实现的，有兴趣的可以自行了解匈牙利算法，里面的数学原理会比较复杂。
```python
class HungarianAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of some components.
    For DETR the costs are weighted sum of classification cost, regression L1
    cost and regression iou cost. The targets don't include the no_object, so
    generally there are more predictions than targets. After the one-to-one
    matching, the un-matched are treated as backgrounds. Thus each query
    prediction will be assigned with `0` or a positive integer indicating the
    ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        match_costs (:obj:`ConfigDict` or dict or \
            List[Union[:obj:`ConfigDict`, dict]]): Match cost configs.
    """

    def __init__(
        self, match_costs: Union[List[Union[dict, ConfigDict]], dict,
                                 ConfigDict]
    ) -> None:

        if isinstance(match_costs, dict):
            match_costs = [match_costs]
        elif isinstance(match_costs, list):
            assert len(match_costs) > 0, \
                'match_costs must not be a empty list.'

        self.match_costs = [
            TASK_UTILS.build(match_cost) for match_cost in match_costs
        ]

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               img_meta: Optional[dict] = None,
               **kwargs) -> AssignResult:
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places. It may includes ``masks``, with shape
                (n, h, w) or (n, l).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                ``labels``, with shape (k, ) and ``masks``, with shape
                (k, h, w) or (k, l).
            img_meta (dict): Image information.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert isinstance(gt_instances.labels, Tensor)
        num_gts, num_preds = len(gt_instances), len(pred_instances)
        gt_labels = gt_instances.labels
        device = gt_labels.device

        # 1. assign -1 by default
        assigned_gt_inds = torch.full((num_preds, ),
                                      -1,
                                      dtype=torch.long,
                                      device=device)
        assigned_labels = torch.full((num_preds, ),
                                     -1,
                                     dtype=torch.long,
                                     device=device)

        if num_gts == 0 or num_preds == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels)

        # 2. compute weighted cost
        cost_list = []
        for match_cost in self.match_costs:
            cost = match_cost(
                pred_instances=pred_instances,
                gt_instances=gt_instances,
                img_meta=img_meta)
            cost_list.append(cost)
        cost = torch.stack(cost_list).sum(dim=0)

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels)
```

### 3. Innovation
首个应用Transformer的端到端检测模型。