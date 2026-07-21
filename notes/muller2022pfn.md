# muller2022pfn — 阅读笔记

**论文：** Transformers Can Do Bayesian Inference (ICLR 2022)
**代码：** `code/TransformersCanDoBayesianInference/`
**GitHub：** https://github.com/automl/TransformersCanDoBayesianInference

---

## 核心思想

用 Transformer 来**摊销贝叶斯推断**——离线在合成数据上预训练，推理时一次前向传播直接输出后验预测分布（PPD），不需要 MCMC 或 VI。

理论保证：最小化 Prior-Data NLL 等价于最小化对真实 PPD 的 KL 散度。

---

## Transformer 架构（`transformer.py`）

### Token 设计

每个训练样本 $(x_i, y_i)$ 变成一个 token：

```
token_i = Linear(x_i) + Linear(y_i)   # 分别线性投影后相加
```

- `encoder`：处理 x 的线性层
- `y_encoder`：处理 y 的线性层（单独一个）
- 两者**相加**而非拼接，与 Transformer 位置编码的做法一致

**为什么相加等价于拼接：**

```
W_x · x + W_y · y  =  [W_x | W_y] · [x; y]
```

参数量相同（都是 2d²），结果相同，只是写法更清晰。

### Attention Mask（`generate_D_q_matrix`）

mask 是**加法 mask**，直接加到 attention score 上：

```
softmax( QKᵀ/√d + mask ) · V
```

- `0.0` → 允许 attend（score 不变）
- `-inf` → 屏蔽（softmax 后变为 0）

矩阵语义（行=谁在问，列=问谁）：

```
         train₀ train₁ train₂ query₀ query₁
train₀ [   0      0      0    -inf   -inf  ]
train₁ [   0      0      0    -inf   -inf  ]
train₂ [   0      0      0    -inf   -inf  ]
query₀ [   0      0      0      0    -inf  ]
query₁ [   0      0      0    -inf     0   ]
```

- train token：只能看 train，看不到 query（防止信息泄露）
- query token：能看所有 train，不能看其他 query，只能看自己
- query 保留对角线是为了防止 softmax 分母全为 0（NaN）

### forward 流程

```python
# 1. 编码
x_emb = encoder(x)          # Linear(features → d)
y_emb = y_encoder(y)        # Linear(1 → d)

# 2. 训练行：x + y 相加
train_tokens = x_emb[:n] + y_emb[:n]

# 3. 测试行：只有 x
src = cat([train_tokens, x_emb[n:]])

# 4. Transformer + mask
output = transformer_encoder(src, mask)

# 5. 只取测试行的输出
return output[n:]
```

---

## Encoder（`encoders.py`）

### Linear encoder（实际用的）

```python
Linear = nn.Linear   # 直接线性投影，适合数值型特征
```

### CanEmb（类别变量 encoder）

```python
class CanEmb(nn.Embedding):
    # 把 d 维平均分给每个特征，各自做 embedding lookup，最后拼接
    embedding_dim = embedding_dim // num_features
```

- `num_features`：特征数量
- `num_embeddings`：每个特征的类别数（假设所有特征类别数相同）
- `embedding_dim`：总输出维度

**实际使用：** 只在 `FewShotOmniglot.ipynb` 里作为 `y_encoder` 使用（5-way 分类，标签有 5 个取值）。对 tabular 的 x 没有使用（类别数不一样，假设太强）。

---

## 训练数据生成（Prior Sampling）

### GP Prior（`priors/gp.py`）

基于**相似性**的模型：相近的 x 对应相近的 y。

```python
x = uniform[0,1] 随机采样
gpr = GaussianProcessRegressor(kernel=RBF(length_scale=0.6), optimizer=None)
y = gpr.sample_y(x)   # 从 GP prior 采样函数值
```

- `optimizer=None`：kernel 参数固定，不训练，直接从 prior 采样
- 适合时序/空间/物理信号等平滑场景

**`evaluate` 函数：** 不参与训练，是用来计算 GP oracle baseline 的——模拟"知道真实 kernel"时的最优预测性能，PFN 的 loss 曲线应该逼近它。

### BNN/MLP Prior（`priors/mlp.py`）

基于**随机神经网络**的模型：数据由某个 MLP 生成。

```
1. 随机采样 MLP 结构（层数、宽度、激活函数随机）
2. 随机初始化权重
3. 从正态分布采样 x
4. x 过随机 MLP + GaussianNoise → y
5. 归一化 x, y，不足 num_features 的补零
```

每个随机 MLP 代表一个"任务"，覆盖各种非线性函数关系，更适合 tabular 数据（列间复杂交互）。

### GP vs BNN 对比

| | GP prior | BNN/MLP prior |
|--|--|--|
| 归纳偏置 | 平滑性、局部相关性 | 层次化非线性函数 |
| x 来源 | uniform[0,1] | 标准正态 |
| y 来源 | GP 函数采样 | 随机 MLP forward |
| 适合场景 | 时序/空间数据 | tabular 数据 |

### `get_batch` 返回三个值

```python
return x, y, y   # (x, y_noisy, y_clean)
```

第三个本应是无噪声的 y，但两个 prior 都没有单独追踪 clean y，所以用同一个 y 填了两次。

---

## 训练循环（`train.py`）

### 数据集

**没有固定的训练集或测试集**——每个 batch 都从 prior 实时生成，永远不会重复：

```python
dl = priordataloader_class(num_steps=steps_per_epoch, batch_size=batch_size, seq_len=bptt)
# 每次迭代调用 get_batch 生成新的合成数据
```

### Training Loop 结构

```python
for epoch in range(epochs):
    for batch, (data, targets) in enumerate(dl):   # 1. 从 prior 拿合成数据
        single_eval_pos = single_eval_pos_gen()     # 2. 随机决定 context/query 切分点
        output = model(data, single_eval_pos=...)   # 3. forward
        targets = targets[single_eval_pos:]         # 4. 只取 query 位置的 y
        losses = criterion(output, targets)         # 5. 算 loss
        loss.backward()                             # 6. 反向传播
        clip_grad_norm_(model.parameters(), 1.)    # 7. 梯度裁剪
        optimizer.step()                            # 8. Adam 更新
    scheduler.step()                                # 9. cosine lr 衰减
```

### 特殊训练技巧

**没有 Curriculum Learning**（这是 TabICL 引入的，PFN 的 N 很小不需要）。

实际用到的技巧：

| 技巧 | 说明 |
|------|------|
| Cosine LR + warmup | 先线性 warmup，再 cosine 衰减 |
| 梯度裁剪 | `clip_grad_norm_(1.0)`，防止梯度爆炸 |
| 梯度累积 | 每 k 个 batch 才更新一次，模拟大 batch |
| Weighted `single_eval_pos` 采样 | 随机变化 context 大小；加权避免小 n 被过度训练 |
| 输出层零初始化 | 每层 Transformer 输出投影初始化为 0，训练初期更稳定 |

---

## 任务类型与 Decoder

### Decoder 结构

```python
decoder = nn.Sequential(
    nn.Linear(d, nhid),
    nn.GELU(),
    nn.Linear(nhid, n_out)
)
```

每个 query 行的 d 维 embedding 过这个 MLP，输出 `n_out` 维，`n_out` 的含义取决于任务。

### 分类任务

```python
n_out = num_classes
criterion = CrossEntropyLoss()
```

输出 `num_classes` 维 logits → softmax → 类别概率。

原始 PFN 的合成数据**只支持二分类**（把连续 y 二值化）。TabPFN v1 才加入真正的多类别支持。

### 回归任务 — Riemann Distribution（主推）

```python
n_out = 100   # 默认 100 个 bin
criterion = BarDistribution(borders=...)
```

把连续 y 离散化成 B 个 bin，输出每个 bin 的概率——**把回归变成分类问题**。

**Bin 边界的设计：等概率划分**

- 训练前从 prior 大量采样 y，找出使每个 bin 概率相等的边界
- 边界一旦确定就**固定不变**，推理时不会根据 context 动态调整
- 等概率保证每个 bin 训练样本数相同，cross-entropy 梯度均匀

```
y 集中在中间时：
|bin1|bin2|  bin3  |   bin4   |
 窄    窄     宽         宽      ← 稀疏区 bin 更宽
```

**为什么边界固定而不动态：** Transformer 输出维度固定（100维），推理时无法改变。PFN 的设计哲学是"y 的分布由 prior 编码，不从 context 推断"。

**`normalize_data` 的实际位置：** 归一化发生在**训练数据生成时**（`priors/mlp.py L177`），不是推理时：

```python
x, y = normalize_data(x), normalize_data(y)   # z-score，生成合成数据后立即执行
```

bin 边界是基于这个归一化后的 y（mean=0, std=1）计算的。推理时对真实数据的 y 并没有显式归一化逻辑——原始 PFN 的回归实验主要在合成场景下验证，真实 tabular 数据的 y 范围问题留给了后续工作处理。TabICL v2 用 999-quantile + pinball loss 从根本上解决了这个问题——bin 边界根据每个任务的 y 分布自适应，不依赖 prior。

**推理时提取预测值：**

```
预测均值 = Σ (bin中心 × 该bin概率)
```

---

## 与后续工作的关系

- **TabPFN v1**：直接应用 PFN 到 tabular 分类；换更真实的 SCM prior；加多类别支持、预处理、ensembling
- **TabPFN v2**：per-cell token + 交替 row/col attention；随机列向量实现架构级列不变性；支持缺失值/类别/回归
- **TabICL**：3-stage pipeline，把列 embedding 和 ICL Transformer 解耦，N 上限扩到 500K
