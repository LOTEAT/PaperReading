<!--
 * @Author: LOTEAT
 * @Date: 2024-10-03 20:45:47
-->
## Attention Is All You Need
- 前置知识：PyTorch
- 作者：Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- [文章链接](https://arxiv.org/pdf/1706.03762)
- [代码链接](https://github.com/harvardnlp/annotated-transformer)
- [视频链接](https://www.bilibili.com/video/BV1pu411o7BE/?spm_id_from=333.337.search-card.all.click&vd_source=962a14fe78f4c5b36a73df66a4d2f23b)

### 1. Introduction
这篇论文李沐老师已经讲解得非常仔细了，推荐去看原视频。

### 2. Code
代码上我选取了Harvard的仓库，但是由于数据上的问题，这个代码可能已经无法执行了。
我们先来看整体的模型：
```python
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
```
这是一个encoder-decoder架构。先来看Encoder。
```python
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```
经过N层layer后，输入再经过一层LayerNorm就获得了Encoder的输出。
```python
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
```
每一层的encoder layer都是有self attention和feedforward组成。
```python
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
```
这就是attention score的计算方法。这里将mask为0所对应的score的位置赋值为-1e9，这样经过softmax后就基本为0了。
```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)
```
在多头自注意力机制中，虽然我们应该是用多个Linear层获得多个query，key和value，但是实际上，我们完全可以通过一个Linear层获得一个query，key和value。然后计算注意力分数。只需要注意对应位置关系即可。
```python
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
```
decoder layer则不太一样，它需要使用两个multi-head attention。
这里需要注意两个mask，一个是src_mask，一个是tgt_mask。tgt_mask只会被用在decoder layer中，而src_mask会被用在encoder和decoder layer中。

先来说src_mask。现在假定输入的batch是32，每个word我们使用一个256维度的向量进行表征，一句话我们最多使用128个词向量进行表征。对于word少于128的句子，我们使用0向量填充，对于word多于128的句子，我们直接截断。那么最终的输入应该是$32\times 128 \times 256$。在预处理中，由于有些不足128word的句子被我们使用0向量进行了填充，那么实际上在计算注意力时，这些word是不应该被我们关注的。src_mask的维度就应该$32\times 128$，这是用来表征哪些词是无关的。

tgt_mask是一个用于遮掩gt的mask。对于gt，我们处理是类似的。只不过每一个word会有一个独立的id去表示，这样我们就可以将seq2seq的任务转化为分类的任务。那么gt的输入就应该是$32\times 128$的，此时gt遮盖了padding的word。而且，考虑到在预测第i个词时，是不应该看到第i+1个词的，那么就可以通过tgt_mask将attention score的上三角部分置为负无穷。

transformer最后的输出，我们可以通过贪心算法进行解码。对于输出的值，我们不断进行解码直至输出到停止符为止

transformer还需要注意的一点就是它的训练过程和推理过程中，所使用的tgt是不一样的。在训练中，tgt就是gt，这是为了计算损失。但是在推理时，tgt应该是一个起始符。