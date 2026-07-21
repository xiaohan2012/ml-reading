# TabICL v1 阅读笔记

**论文：** TabICL: A Tabular Foundation Model for In-Context Learning on Large Data (qu2025tabicl, ICML 2025)
**arXiv：** 2502.05564
**代码：** `soda-inria/tabicl` tag `v0.1.4`
**LaTeX：** `wiki/raw/2502.05564/TabICL.tex`

---

## 一、核心思想与 Motivation

### 问题：TabPFNv2 扩不了规模

TabPFNv2 的交替 row/col attention 复杂度是 $O(N^2 M + NM^2)$

- $N = 10\text{K}$ 是 v2 的实际上限，超了就 OOM 或慢得不实用
- 真实工业数据集动辄百万行，v2 完全不可用

**根本原因**：v2 把每个格子 $(行, 列)$ 都保持为独立 token，从头到尾都是 $(N, M, d)$ 的三维张量，列维度永远不消失，每层 attention 都要对 $N$ 和 $M$ 两个轴做计算。

### 核心 insight：先把列维度消掉

TabICL 的根本思路：**先把每行压缩成一个固定大小的向量，然后再做 ICL**。

压缩之后，ICL 阶段只需要对 $(N, d)$ 做 attention，复杂度从 $O(N^2 M)$ 降到 $O(N^2)$——$M$ 这个因子彻底消失。

这就是整个架构的出发点：**先 embed，再 ICL**，而不是 v2 的"边 embed 边 ICL 交替做"。

### 三段流水线

```
原始表格 (N, M)
  ↓ TF_col — 列级 Set Transformer（对每列的 N 个值做 attention）
(N, M, d)
  ↓ TF_row — 行级 Transformer + 4 个 CLS token（对每行的 M 个特征做 attention）
(N, 4d=512)   ← 列维度在这里消失
  ↓ TF_icl — 标准 Transformer ICL
输出预测
```

每个阶段只运行**一次**，不像 v2 是 $L$ 层交替循环。

| 阶段 | 输入形状 | 复杂度 | 作用 |
|------|----------|--------|------|
| $\text{TF}_{\text{col}}$ | 每列 $N$ 个值 | $O(NkM)$，$k=128$ 固定 | 学列内分布规律，输出 per-cell 调制系数 |
| $\text{TF}_{\text{row}}$ | 每行 $M$ 个特征 | $O(M^2 N)$，$M \ll N$ | 捕捉特征间相关性，CLS 聚合成固定向量 |
| $\text{TF}_{\text{icl}}$ | $N$ 个行向量 $(N, 512)$ | $O(N^2)$ | ICL，行之间互相 attend |
| **总计** | | $O(M^2 N + N^2)$ | vs v2 的 $O(N^2 M + NM^2)$ |

---

## 二、模型架构

### 顶层调用链

[`tabicl.py:110-188`](https://github.com/soda-inria/tabicl/blob/v0.1.4/src/tabicl/model/tabicl.py#L110)

```python
self.col_embedder   = ColEmbedding(embed_dim=128, num_blocks=3, num_inds=128,
                                   reserve_cls_tokens=4, ...)
self.row_interactor = RowInteraction(embed_dim=128, num_blocks=3, num_cls=4,
                                     rope_base=100000, ...)
icl_dim = 128 * 4  # = 512
self.icl_predictor  = ICLearning(d_model=512, num_blocks=12, ...)
```

两点值得注意：
- `reserve_cls_tokens=4` 传给 `col_embedder`——CLS slot 在 TF_col 阶段就预留，不是 TF_row 里插入
- 维度升宽（128→512）发生在两个模块的边界处，不在任何模块内部

`_train_forward` 核心两行：

```python
representations = self.row_interactor(
    self.col_embedder(X, train_size=train_size)
)   # (B,T,M) → (B,T,M+4,128) → (B,T,512)
out = self.icl_predictor(representations, y_train=y_train)
```

`train_size` 传给 `col_embedder` 是为了让 ISAB 的 MAB₁ 只看训练行——防数据泄露。

---

### TF_col：列级 Set Transformer + FiLM

[`embedding.py:59`](https://github.com/soda-inria/tabicl/blob/v0.1.4/src/tabicl/model/embedding.py#L59)

#### 设计问题

直觉上用"每列专属 embedding 模块"，但这样做有问题：每列的权重是预训练时学死的，跨表无法迁移。

TabICL 的解法是把 feature embedding 改造成一个 **set-input 问题**：
- 输入：一列的所有 $N$ 个值（无序集合）
- 输出：$N$ 个 per-cell 的 embedding 向量

#### 形状变换

[`embedding.py:147`](https://github.com/soda-inria/tabicl/blob/v0.1.4/src/tabicl/model/embedding.py#L147)

```python
X = pad(X, (4, 0), value=-100.0)           # (B,T,M) → (B,T,M+4)，前4列CLS占位
features = X.transpose(1,2).unsqueeze(-1)  # → (B,M+4,T,1)，每列成独立序列
embeddings = self._compute_embeddings(features, train_size)  # (B,M+4,T,128)
return embeddings.transpose(1,2)           # → (B,T,M+4,128)
```

关键是 `transpose`：把列轴放到 batch 维，让 $M+4$ 列并行送进 ISAB。CLS slot 填 `-100.0`，`SkippableLinear` 和 ISAB 看到就跳过。

#### ISAB（Induced Self-Attention Block）

[`layers.py:470`](https://github.com/soda-inria/tabicl/blob/v0.1.4/src/tabicl/model/layers.py#L470)

用 Set Transformer 处理，$k=128$ 个 inducing vectors 把复杂度从 $O(N^2)$ 降到 $O(Nk)$：

$$U = \text{Lin}(c) \in \mathbb{R}^{N \times d}$$
$$M = \text{MAB}_1(V_I,\ U_{\text{train}},\ U_{\text{train}}) \in \mathbb{R}^{k \times d}$$
$$V = \text{MAB}_2(U,\ M,\ M) \in \mathbb{R}^{N \times d}$$

- **$U$ 的含义**：$c \in \mathbb{R}^N$ 是一整列所有行（训练 + 测试），$U = \text{Lin}(c)$ 包含全部 $N$ 行；$U_{\text{train}}$ 是 $U$ 的上半部分（训练行切片），两者共享同一次线性投影
- **MAB₁**：$k$ 个 inducing vectors $V_I$ 作为 Q，**只对 $U_{\text{train}}$** 做 attention，得到 $k$ 个摘要向量 $M$——$M$ 只由训练集决定
- **MAB₂**：完整的 $U$（训练 + 测试）作为 Q，对 $M$ 做 attention，把训练集摘要广播回每一行（含测试行）
- **关键**：测试行能看到训练集的分布信息（通过 $M$），但 $M$ 本身不被测试集影响 → 防止数据泄露

代码：

```python
# layers.py:557-562
ind_vectors = self.ind_vectors.expand(*batch_shape, 128, 128)
hidden = self.multihead_attn1(ind_vectors, src[..., :train_size, :], src[..., :train_size, :])
out    = self.multihead_attn2(src, hidden, hidden)
```

#### FiLM 调制输出 / Hypernetwork

[`embedding.py:118`](https://github.com/soda-inria/tabicl/blob/v0.1.4/src/tabicl/model/embedding.py#L118)

```python
src     = self.in_linear(features)        # (B,M+4,T,1) → (B,M+4,T,128)
src     = self.tf_col(src, train_size)    # ISAB × 3
weights = self.ln_w(self.out_w(src))      # (B,M+4,T,128)
biases  = self.ln_b(self.out_b(src))      # (B,M+4,T,128)
embeddings = features * weights + biases  # 广播：(B,M+4,T,1) × (B,M+4,T,128)
```

$$e_j = W_j \odot c_j + B_j$$

- TF_col 输出的是**仿射调制系数** $(W, B)$，而不是直接的 embedding 向量
- 最终的 cell 表示 = 原始标量值 × 学到的 weight + 学到的 bias
- 好处：cell 表示永远依赖原始值，模型学的是"如何根据列分布特性来 embed 值"，而不是特定列的语义
- **两种叫法，同一个计算图**：
  - 叫 **Hypernetwork**（论文原文用词）：强调 TF_col 在"生成另一个网络的参数"——外层网络输出 $(W, B)$，内层网络是 $W \odot c + B$
  - 叫 **FiLM**（CV 领域叫法）：强调输出形式是仿射 $W \odot x + B$，由条件信号（列分布）控制

线性层维度：

| 层 | 权重形状 | 作用 |
|---|---|---|
| `in_linear` | $(128, 1)$ | 标量升维 |
| `out_w` | $(128, 128)$ | 生成 per-cell weight $W$ |
| `out_b` | $(128, 128)$ | 生成 per-cell bias $B$ |

论文可视化（Figure 3）显示，ISAB 的 inducing vector 表示在 PCA 空间里按**偏度和峰度自然聚类**——TF_col 学到的是分布属性。

#### Batch 里不同特征数的处理

训练时同一 batch 里不同表可能有不同特征数，pad 到最大列数 HC：

```python
mask = indices < d.unsqueeze(1)           # (B, HC)，True = 真实列
features = X[mask].unsqueeze(-1)          # 只取真实列送进 _compute_embeddings
embeddings = torch.zeros(B, HC, T, 128)
embeddings[mask] = effective_embeddings   # 结果填回，pad 列保持零向量
```

注意两种 mask 的区别：
- **`skip_mask`（SkippableLinear 里）**：检测 CLS slot 的 `-100` 占位符
- **`mask`（`_train_forward` 里）**：检测特征数不足的 pad 列

---

### TF_row：行级 Transformer + CLS 聚合

[`interaction.py:53`](https://github.com/soda-inria/tabicl/blob/v0.1.4/src/tabicl/model/interaction.py#L53)

#### 为什么是 4 个 CLS token 而不是 1 个

这是内存和表达能力之间的权衡：

- TF_col 和 TF_row 内部用 $d=128$ 来省内存（TF_row 要对每行 $M$ 个 token 做 attention，内存和 $M$ 成正比）
- 但 TF_icl 需要足够丰富的行表示，128 维可能不够
- 4 个 CLS token 拼接得到 $4 \times 128 = 512$ 维，是一种**廉价的升维方式**：多了 4 个 token 但 $M \gg 4$，TF_row 计算量基本不变，却把传给 TF_icl 的向量维度扩大了 4 倍
- 本质是把"宽网络"的成本从 TF_row 内部 attention 转移到边界处的拼接操作上，代价极小

#### 核心计算

[`interaction.py:89`](https://github.com/soda-inria/tabicl/blob/v0.1.4/src/tabicl/model/interaction.py#L89)

```python
# 覆盖 CLS slot（-100 占位符 → 可学习参数）
cls_tokens = self.cls_tokens.expand(B, T, 4, 128)
embeddings[:, :, :4] = cls_tokens

# 跑 TF_row（3 层 Transformer + RoPE）
outputs = self.tf_row(embeddings)          # (B,T,M+4,128)

# 取前 4 个 CLS 输出，拼接成行向量
cls_outputs = outputs[..., :4, :]         # (B,T,4,128)
del outputs                                # 立刻释放显存
return cls_outputs.flatten(-2)            # (B,T,512)
```

M 个 feature token 的输出直接丢弃——它们的作用是在 attention 里把信息汇聚到 CLS token 上。

`key_mask` 屏蔽 pad 列，防止真实 feature token attend 到零向量：

```python
key_mask = indices >= d.view(B, 1, 1)    # True = pad 列，attention 时加 -inf
```

#### 表示坍塌问题

如果两列的分布完全一样（比如 `balance scale` 数据集里 4 个特征都是同一个离散分布），TF_col 输出的 embedding 几乎一样，TF_row 就无法区分"这是第 0 列"还是"这是第 1 列"。

- **v2 的解法**：每次 forward 用随机向量标记列身份（random attribute tokens）
- **TabICL 的解法**：在 TF_row 里用 **RoPE（旋转位置编码）**，按列 index 旋转 Q/K，不同列产生不同旋转角度

#### RoPE 原理

RoPE 不把位置信息加到向量上，而是把位置信息**编码进 attention 的计算过程**里：在计算 $Q \cdot K^\top$ 之前，先对 Q 和 K 按位置做旋转：

$$\text{score}(p, p') = (R_p \mathbf{q})^\top (R_{p'} \mathbf{k}) = \mathbf{q}^\top R_{p'-p} \mathbf{k}$$

旋转矩阵相乘后只剩**相对位置差** $p' - p$，attention score 只取决于两 token 之间的距离，不依赖绝对位置。

- $R_p$：位置 $p$ 对应的旋转矩阵，实际是 $d_{\text{head}} \times d_{\text{head}}$（TabICL 里 16×16）的**分块对角矩阵**，沿对角线排 $d/2$ 个独立的 2×2 旋转块，每块用不同频率 $\theta_i$：

$$R_p = \begin{pmatrix} \cos(p\theta_0) & -\sin(p\theta_0) & & \\ \sin(p\theta_0) & \cos(p\theta_0) & & \\ & & \cos(p\theta_1) & -\sin(p\theta_1) \\ & & \sin(p\theta_1) & \cos(p\theta_1) \\ & & & & \ddots \end{pmatrix}$$

  代码里不真的构造这个大矩阵，而是直接对向量按 2 维一对操作（`rotate_half` + cos/sin 逐元素乘），效果等价但更省内存

- $R_{p'}$：同理，作用在位置 $p'$ 的 K 上
- $R_{p'-p}$：来自旋转矩阵的乘法性质——两个旋转相乘等于角度相加，所以 $R_p^\top \cdot R_{p'} = R_{p'-p}$；展开后 $(R_p \mathbf{q})^\top(R_{p'}\mathbf{k}) = \mathbf{q}^\top R_{p'-p} \mathbf{k}$，结果只跟列间距 $p'-p$ 有关，和各自绝对位置无关

TabICL 里的"位置"是**列 index**（0, 1, 2, …），base=100000（比 LLaMA 的 10000 大 10 倍，让旋转在列数超过训练时上限 100 时也不会角度溢出，增强泛化能力）。

RoPE 插入点在 [`attention.py:154-156`](https://github.com/soda-inria/tabicl/blob/v0.1.4/src/tabicl/model/attention.py#L154)：

```python
q, k, v = F._in_projection_packed(query, key, value, ...)  # 线性投影
q = q.view(..., num_heads, head_dim).transpose(...)         # reshape 成多头
k = k.view(..., num_heads, head_dim).transpose(...)
if rope is not None:
    q = rope.rotate_queries_or_keys(q)   # 对 Q 按列 index 旋转
    k = rope.rotate_queries_or_keys(k)   # 对 K 按列 index 旋转
# 然后才计算 Q·Kᵀ
```

RoPE 的代价：对列顺序敏感（交换两列结果不同），破坏了置换不变性。
补救：推理时对列顺序做 ensemble（Latin square 置换），近似恢复置换不变性。

#### 为什么选 RoPE 而不是随机向量（论文的论证）

论文对 RoPE 的论证相当简短：实验发现有 collapse 问题，RoPE 能缓解，TabPFN v2 用随机向量，TabICL 选择了"不同的策略"。没有直接论证 RoPE 比随机向量更好。

附录里有一个更深的视角：RoPE 的低维度分量旋转快（像随机噪声），高维度分量旋转慢（携带语义），所以 RoPE 本质上是"以可控、可预测、可泛化的方式给每个特征引入噪声作为标识符"——和随机向量的思路其实很接近，都是引入噪声来区分列。

**两种方法的实际差异**（来自附录 ensemble size 实验）：

- TabPFN v2 在 ensemble size ≤4 时就有明显提升
- TabICL 需要 ensemble size ≥8 才有明显效果

原因：TabPFN v2 从不消除列维度，列 shuffle 对输出影响更大，ensemble 成员之间相关性更低，收益更快。TabICL 列维度在 TF_row 后已压缩，shuffle 对 TF_icl 影响较小，成员之间更相关，需要更多成员。

**结论**：论文没有明确说哪个更好。但随机向量在 ensemble 效率上略占优势，也更干净（架构级别保证置换不变性）；RoPE 引入了列间距信息，但这个信息在 tabular 数据里本身没有语义意义。

---

### TF_icl：ICL 阶段

[`learning.py:52`](https://github.com/soda-inria/tabicl/blob/v0.1.4/src/tabicl/model/learning.py#L52)

```python
self.tf_icl    = Encoder(d_model=512, num_blocks=12, nhead=4, use_rope=False, ...)
self.y_encoder = OneHotAndLinear(max_classes=10, d_model=512)
self.decoder   = nn.Sequential(Linear(512, 1024), GELU(), Linear(1024, 10))
```

核心计算 [`learning.py:204`](https://github.com/soda-inria/tabicl/blob/v0.1.4/src/tabicl/model/learning.py#L204)：

```python
R[:, :train_size] += self.y_encoder(y_train.float())  # label 注入训练行
src = self.tf_icl(R, attn_mask=train_size)             # ICL masking（整数形式）
out = self.decoder(src)                                # (B,T,10)
out = out[:, train_size:]                              # 只取测试行输出
```

**`attn_mask=train_size` 整数 mask**：训练行互相 attend，测试行只 attend 训练行，在 [`attention.py:162`](https://github.com/soda-inria/tabicl/blob/v0.1.4/src/tabicl/model/attention.py#L162) 里用两段独立 attention 实现，比构造完整 mask 矩阵更省内存。

**超过 10 类**：递归建 classification tree，每层 ≤10 个 super-class，最终概率 = 路径上各层概率之积。不在论文架构图里，是推理时的工程手段。

---

## 三、预训练数据生成

两个 SCM 生成器按 **70/30** 混合：

**`MLPSCM`（70%）**：沿用 TabPFN v1 思路，随机 MLP 作为 DAG，$X$ 和 $y$ 从 MLP 的中间隐变量里随机抽取——两者共同的上游隐变量决定，更符合真实 tabular 数据的因果结构。

**`TreeSCM`（30%）**：把 MLP 里的线性层换成树模型（XGBoost / RandomForest / ExtraTrees）：

```python
y_fake = random noise                  # 随机目标
model.fit(X_input, y_fake)            # 用随机目标拟合树模型
output = model.predict(X_input)       # 预测值作为这层输出
```

目的：引入树模型特有的分段常数、非连续函数，让合成数据的函数形态更接近真实 tabular 数据。

**激活函数多样化**：19 种激活函数（vs TabPFN v1 的 4 种），包括 sine、Gaussian、step function、GP 采样函数等非单调/不连续函数。

**与 TabPFN v2 的关系**：框架完全一样，TabICL 是 v2 的超集——同一个 SCM 框架上，加了 TreeSCM + 更多激活函数 + Curriculum。

---

## 四、训练流程

### Curriculum 两个维度

论文描述的三阶段 curriculum 在代码里是**三次独立的训练运行**，每次用不同的命令行参数配置：

| 阶段 | `max_seq_len` | 步数 | 冻结模块 |
|------|--------------|------|---------|
| 1 | 1,024 | 160K | 无 |
| 2 | 40,000（log-uniform 采样） | 2K | 无 |
| 3 | 60,000 | 50 | TF_col + TF_row |

训练主循环 [`run.py:413`](https://github.com/soda-inria/tabicl/blob/v0.1.4/src/tabicl/train/run.py#L413) 是**单一循环**，没有阶段切换逻辑。Curriculum 完全靠外部脚本控制参数实现。

### `adjust_max_features`

[`dataset.py:242`](https://github.com/soda-inria/tabicl/blob/v0.1.4/src/tabicl/prior/dataset.py#L242)

行数越多，特征数上限自动越低，在固定显存预算下动态权衡行数和特征数：

```python
if seq_len <= 10240:  return min(100, max_features)
if seq_len <= 20000:  return min(80,  max_features)
if seq_len <= 30000:  return min(60,  max_features)
if seq_len <= 40000:  return min(40,  max_features)
if seq_len <= 50000:  return min(30,  max_features)
if seq_len <= 60000:  return min(20,  max_features)
```

### 冻结模块

[`run.py:197`](https://github.com/soda-inria/tabicl/blob/v0.1.4/src/tabicl/train/run.py#L197)

```python
if self.config.freeze_col:
    model.col_embedder.eval()
    for param in model.col_embedder.parameters():
        param.requires_grad = False
```

阶段 3 冻结 TF_col 和 TF_row 的原因：60K 行的大表训练不稳定，只让 TF_icl 适应大规模 ICL 场景，避免破坏前两阶段已学好的 embedding。

---

## 五、推理 Ensemble

### 三个维度

[`preprocessing.py:808`](https://github.com/soda-inria/tabicl/blob/v0.1.4/src/tabicl/sklearn/preprocessing.py#L808)

```python
configs = list(itertools.product(shuffle_shift_configs, norm_methods))[:n_estimators]
```

| 维度 | 内容 |
|---|---|
| **列顺序** | Latin square 置换 |
| **类别偏移** | 类别标签循环偏移，消除类别顺序偏置 |
| **归一化方式** | standard scale、RTDL quantile 等 |

默认 32 个成员，预测取平均（softmax temperature=0.9）。

### Latin square

[`preprocessing.py:772`](https://github.com/soda-inria/tabicl/blob/v0.1.4/src/tabicl/sklearn/preprocessing.py#L772)

$M \times M$ 的矩阵，每行每列都是 $\{0,...,M-1\}$ 的一个置换。每列恰好在每个位置出现一次，取平均后每列对每个位置的贡献完全均等，RoPE 引入的位置偏置被彻底平均掉。比纯随机 shuffle 更系统、更均匀。

### TF_col 只跑一次的速度技巧

[`embedding.py:260`](https://github.com/soda-inria/tabicl/blob/v0.1.4/src/tabicl/model/embedding.py#L260)

```python
# 只对第一张表跑一次 TF_col
first_embeddings = col_embedder(X[0])            # (M+4, T, 128)

# 其余成员直接对 embedding 做列轴重排
embeddings[i] = first_embeddings[mapping[i]]    # 不重新计算
```

TF_col 列间独立，shuffle 列顺序等价于 shuffle embedding 的列轴。TF_col 是参数最多的模块，只跑一次，TF_row 和 TF_icl 才跑 32 次。

---

## 六、与 TabPFN v2 的对比

| 方面 | TabPFNv2 | TabICL v1 |
|------|----------|-----------|
| **Token 粒度** | per-cell $(N, M, d)$ 端到端 | 三段流水线，列维度在 TF_row 后消失 |
| **复杂度** | $O(N^2 M + NM^2)$ | $O(M^2 N + N^2)$ |
| **$N$ 上限** | ~10K | ~500K |
| **列身份** | 随机向量（每次 forward 重新采样，Linear 投影可学习） | RoPE（按列 index 旋转，确定性） |
| **置换不变性** | 架构级别保证 | RoPE 破坏，靠 ensemble 近似恢复 |
| **Ensemble 效率** | ≤4 成员即有明显收益 | 需要 ≥8 成员才有明显收益 |
| **列间距信息** | 无 | 有（但在 tabular 数据里无实际语义） |
| **feature embedding** | 共享 `Linear(1→d)` + 随机列向量 | Set Transformer + FiLM 调制系数 |
| **预训练数据** | SCM prior | SCM + 树基 SCM（70/30），19 种激活函数 |
| **任务** | 分类 + 回归（两个 checkpoint） | 仅分类（v1） |
| **训练代码** | 未开源 | 开源 |
