# TabPFN v2 阅读笔记

**论文：** Accurate Predictions on Small Data with a Tabular Foundation Model (hollmann2025tabpfnv2, Nature 2025)
**代码：** `PriorLabs/TabPFN` tag `v2.0.9_`

---

## 一、核心思想

v0/v1 和 v2 共享 PFN 框架：离线在合成数据上预训练，推理时一次前向传播直接输出后验预测分布。

v2 的动机是修复 v0/v1 的根本缺陷：**一行一 token + 学死的列权重**。

v0/v1 的所有限制都来自同一个根因：

| 限制 | 原因 |
|------|------|
| 列顺序偏置 | $W_x[:, j]$ 在预训练时学死，每列权重固定 |
| $M \le 100$ 硬上限 | `Linear(100 → d)` 固定维度 |
| 无法处理 NaN / 类别特征 | 行压缩前需要外部 imputation |
| v1 只能分类（≤10 类） | decoder 输出 10 维 logits |

v2 把根因改掉（per-cell token + 随机列身份），其他问题自然跟着解决。

---

## 二、模型架构

### Token 粒度

$$\text{v0/v1: } (N, d) \quad\longrightarrow\quad \text{v2: } (N, M, d)$$

每个 $(行, 列)$ 格子都是一个独立的 token。

**格子如何变成向量：**

每个格子的原始值是标量，用 [`LinearInputEncoderStep`](https://github.com/PriorLabs/TabPFN/blob/v2.0.9_/src/tabpfn/model/encoders.py#L395)（即 `Linear(1 → d)`）投影：

$$\mathbf{e} = W \cdot x + \mathbf{b}, \quad W \in \mathbb{R}^{d \times 1},\quad \mathbf{b} \in \mathbb{R}^d$$

- 所有格子**共享同一个** `Linear(1 → d)`——$W$ 只有一列，所有格子投影方向相同，magnitude 不同
- 区分不同列靠的是后面加的随机列身份向量，而不是 encoder 本身
- v0/v1 用 `Linear(F → d)`，一行所有列一起投影，每列有专属权重

调用处：[`transformer.py:589-598`](https://github.com/PriorLabs/TabPFN/blob/v2.0.9_/src/tabpfn/model/transformer.py#L589)

```python
embedded_x = einops.rearrange(
    self.encoder(x, ...),
    "s (b f) e -> b s f e",   # encoder 内部把 B*F 合并批处理，出来再拆回 (B,S,F,d)
    b=batch_size,
)
```

### Delta 1：交替 attention

代价是 attention 复杂度从 $O(N^2)$ 升到 $O(N^2 M + NM^2)$，v2 用**交替 attention** 来分解：

```
每一层 (B, S, F, d)：
  ↓  列-attention（在每行内，F 个 token 互相 attend）  → 捕捉特征间相关性
  ↓  行-attention（在每列内，S 个样本互相 attend）     → ICL 信号（哪些训练样本与测试样本相似）
  ↓  MLP
```

- **列-attention**：输入 `(B, S, F, d)`，F 已在 dim -2，直接做 MHA，行之间互不干扰
- **行-attention**：`transpose(1,2)` → `(B, F, S, d)`，S 到 dim -2，做 MHA，再 transpose 回来

两种 attention 轮流叠 $L$ 层，行列之间的信息交叉通过层深度完成，**不存在 $(NM)^2$ 的全连接 attention**。

模块声明：[`layer.py:182-220`](https://github.com/PriorLabs/TabPFN/blob/v2.0.9_/src/tabpfn/model/layer.py#L182)

```python
self.self_attn_between_features = MultiHeadAttention(...)   # 列-attention
self.self_attn_between_items = MultiHeadAttention(
    share_kv_across_n_heads=nhead,   # multiquery：所有 head 共享 K/V
    ...
)
```

sublayer 组装与执行：[`layer.py:401-455`](https://github.com/PriorLabs/TabPFN/blob/v2.0.9_/src/tabpfn/model/layer.py#L401)

```python
sublayers = [attn_between_features, attn_between_items, mlp]
for sublayer, layer_norm in zip(sublayers, self.layer_norms):
    state = sublayer(state)    # 残差在内部（add_input=True）
    state = layer_norm(state)  # post-norm
```

### Delta 2：随机列身份（Randomized Attribute Tokens）

per-cell token 的问题：同一行里 M 个 token 值都是标量，列-attention 无法区分"我是第 3 列"还是"我是第 7 列"。

三种选择：

| 方案 | 结果 |
|------|------|
| 不给身份 | 列-attention 退化，特征无法区分 |
| 学死的 embedding（v0/v1 思路） | 预训练过拟合特定列语义，schema 偏置 |
| **每次 forward 重新随机采样（v2）** | 预训练无法学到"第 0 列 = 某语义"，schema 不变性成为架构属性 |

**`subspace` 实现**（[`transformer.py:729-736`](https://github.com/PriorLabs/TabPFN/blob/v2.0.9_/src/tabpfn/model/transformer.py#L729)）：

$$\mathbf{e}_j = W_{\text{lift}} \cdot \boldsymbol{\epsilon}_j, \quad \boldsymbol{\epsilon}_j \sim \mathcal{N}(0, I_{d/4})$$

```python
embs = torch.randn((F, d // 4))                              # 每次 forward 重新采样
embs = self.feature_positional_embedding_embeddings(embs)    # Linear(d/4 → d)，方向是学的
x += embs[None, None]                                        # 广播到所有行、所有 batch
```

- **随机**：预训练无法记住"第 0 列 = 某语义"
- **有区别**：同一次 forward 里不同列的向量不同，列-attention 能区分
- **Linear 是学的**：学的是"如何把随机噪声映射到有用的方向空间"

结果：**schema 不变性从统计近似（v0/v1 的 ensembling 补救）变成架构属性**。

### Encoder / Decoder

`_forward` 完整数据流（[`transformer.py:420`](https://github.com/PriorLabs/TabPFN/blob/v2.0.9_/src/tabpfn/model/transformer.py#L420)）：

```
x (S,B,F) → reshape (B,S,F,1) → encoder → (B,S,F,d)
y (S,B)   → NaN mask → y_encoder → (B,S,1,d)
                                        ↓
                        add_embeddings（随机列身份）
                                        ↓
                        cat → (B,S,F+1,d)   ← y 拼到最后一列
                                        ↓
                        LayerStack（L × [列-attn → LN → 行-attn → LN → MLP → LN]）
                                        ↓
                        [:, 测试行, -1, :] → decoder → 预测值
```

- **y_encoder**：`NanHandlingEncoderStep`（生成 NaN 指示通道）+ `Linear(2 → d)`，测试行的 y 强制设为 NaN，模型靠 NaN 指示区分训练行和测试行
- **decoder**：`Linear(d → nhid) → GELU → Linear(nhid → n_out)`，结构与 v0/v1 相同，`n_out` 由任务类型决定

### KV Cache + Multiquery

测试样本很多时需要分批预测，训练集 K/V 重复计算是浪费：

```
有 cache：
  第 1 批测试：训练集 K/V 算一遍，存 cache
  第 2 批测试：读 cache K/V，只算测试批的 Q    ← 不重复
```

Multiquery（`share_kv_across_n_heads=nhead`）：所有 head 共享同一组 K/V，cache 内存从 $H\times$ 降到 $1\times$。

---

## 三、任务类型与输出

v2 同时支持分类和回归，靠加载不同的预训练 checkpoint（`task_type="multiclass"` 或 `"regression"`），架构完全相同，只有 `n_out` 和损失函数不同。

| `max_num_classes` | 任务 | `n_out` | 损失函数 |
|---|---|---|---|
| 2 | 二分类 | 1 | BCEWithLogitsLoss |
| >2（如 10） | 多分类 | `max_num_classes` | CrossEntropyLoss |
| 0 | 回归 | `BarDistribution.num_bars` | FullSupportBarDistribution |

`decoder_dict` 设计上支持多个 head（multitask），但实际部署只用了 `"standard"` 一个 key。

---

## 四、Inference

### v0/v1

几乎没有预处理，直接一次 forward pass。v1 的 ensembling（列/类别 shift）是消架构偏置的必要补丁。

### v2 的三层结构

**第一层：预处理**

**为什么需要预处理：** 模型在合成数据（SCM/BNN prior）上训练，合成数据的特征值分布比较"干净"（大致正态、无极端偏斜、无量纲差异）。真实 tabular 数据经常有长尾分布（收入、房价）、不同量纲、极端异常值。预处理把真实数据的分布往合成数据的分布靠拢，减小 distribution gap，让模型能更好地泛化。本质是用预处理弥补 prior 和现实之间的差距。

默认两种预处理配置（以分类为例）：

```
配置1：quantile_uni_coarse + SVD + 类别特征 ordinal 编码 + 保留原始特征
配置2：无变换 + 类别特征当数值处理
```

**第二层：Ensembling**

不同的预处理配置本身就是 ensemble 的成员，取平均降低预处理选择带来的方差。不再是消偏置的补丁。

**第三层：三种 InferenceEngine**

| Engine | 缓存什么 | 适用场景 |
|--------|---------|---------|
| `InferenceEngineOnDemand` | 无 | 内存紧张 |
| `InferenceEngineCachePreprocessing` | 预处理结果 | 多次预测同一训练集 |
| `InferenceEngineCacheKV` | 预处理 + KV | 测试样本分批，追求速度 |

---

## 五、v0/v1 vs v2 核心差异

| 方面 | v0/v1 | v2 |
|------|-------|----|
| **Token 粒度** | 一行一 token $(N, d)$ | 一格一 token $(N, M, d)$ |
| **Attention** | 行-only，$O(N^2)$ | 交替行+列，$O(N^2M + NM^2)$ |
| **列身份** | 学死的 $W_x[:, j]$（预训练过拟合列语义） | 每次 forward 随机采样 |
| **Schema 不变性** | 统计近似（ensembling 补救） | 架构级别 |
| **NaN / 类别** | 外部 imputation | 内置 encoder step（per-cell） |
| **任务** | v0：回归+分类；v1：分类（≤10 类） | 分类 + 回归（两个独立 checkpoint） |
| **样本量上限** | v0：极小（bptt=10）；v1：$N \le 1000$ | $N \le 10\text{K}$ |
| **Ensembling 目的** | 消除架构偏置（必须做） | 降低预处理方差（锦上添花） |
| **预处理** | 无（v1 有简单归一化） | 完整 pipeline（quantile/power/SVD 等） |
| **推理加速** | v1：efficient_eval_masking | KV cache + multiquery test attention |
| **训练数据量** | v0：200万；v1：~2600万 | ~1.3亿合成数据集 |
| **训练代码** | 开源 | 未开源 |
