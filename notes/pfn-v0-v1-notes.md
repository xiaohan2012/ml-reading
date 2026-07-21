# PFN v0 & TabPFN v1 阅读笔记

**论文：**
- v0：Transformers Can Do Bayesian Inference (muller2022pfn, ICLR 2022)
- v1：TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second (hollmann2023tabpfnv1)

**代码：**
- v0：`code/TransformersCanDoBayesianInference/`
- v1：`code/TabPFN/`（tag: v1.0.0）

---

## 一、核心思想

用 Transformer 来**摊销贝叶斯推断**：离线在合成数据上预训练，推理时一次前向传播直接输出后验预测分布（PPD），不需要 MCMC 或变分推断。

理论保证：最小化 Prior-Data NLL 等价于最小化对真实 PPD 的 KL 散度。

v0 和 v1 共享这个核心思想，v1 是 v0 的增量版本，没有根本性的架构变化。

---

## 二、架构骨架

### Token 设计

每个样本行变成一个 token：

$$\mathbf{t}_i = W_x \mathbf{x}_i + W_y y_i$$

```
token_i = Linear(x_i) + Linear(y_i)
```

x 和 y 分别用独立的线性层投影后**相加**，而非拼接。相加与拼接在数学上等价：

$$W_x \mathbf{x} + W_y y = [W_x \mid W_y] \begin{bmatrix} \mathbf{x} \\ y \end{bmatrix}$$

```
W_x · x + W_y · y  =  [W_x | W_y] · [x; y]
```

参数量相同，结果相同。

测试行（query）只有 x，没有 y：

$$\mathbf{t}_i^{\text{query}} = W_x \mathbf{x}_i \quad (\text{no } y)$$

```python
train_tokens = x_emb[:n] + y_emb[:n]   # 训练行：x + y
src = cat([train_tokens, x_emb[n:]])    # 测试行：只有 x
```

### Attention Mask

mask 是**加法 mask**，直接加到 attention score 上（0 = 允许，-inf = 屏蔽）：

```
         train₀ train₁ train₂ query₀ query₁
train₀ [   0      0      0    -inf   -inf  ]
train₁ [   0      0      0    -inf   -inf  ]
train₂ [   0      0      0    -inf   -inf  ]
query₀ [   0      0      0      0    -inf  ]   ← 对角线保留（防 softmax NaN）
query₁ [   0      0      0    -inf     0   ]
```

- train token：只看 train，不看 query
- query token：看所有 train，不看其他 query，只看自己（对角线）

v1 引入了 **efficient_eval_masking**：不预先生成 $(n+m)^2$ 的 mask 矩阵，而是把 `single_eval_pos` 作为整数传给 layer，在 layer 内部分成两次独立 attention 调用：

$$\mathbf{H}_{\text{train}} = \text{Attn}(\mathbf{T}, \mathbf{T}, \mathbf{T}), \quad \mathbf{H}_{\text{query}} = \text{Attn}(\mathbf{Q}, \mathbf{T}, \mathbf{T})$$

```python
src_left  = attn(train, train, train)   # train self-attention，大小 n²
src_right = attn(query, train, train)   # query cross-attention，大小 m×n
```

内存从 $O((n+m)^2)$ 降到 $O(n^2 + nm)$，同时去掉了 v0 中 query 的自注意力对角线（语义上更干净）。

### Encoder / Decoder

- **x encoder**：$\mathbf{e}_x = W_x \mathbf{x}$，`nn.Linear(num_features → emsize)`
  - v1 新增 NaN 处理：`replace_nan_by_zero=True`，或 `NanHandlingEncoder`（特征数翻倍，加 NaN 指示通道）
- **y encoder**：$\mathbf{e}_y = W_y y$，`nn.Linear(1 → emsize)`
- **Decoder**：$\hat{y} = W_2\,\text{GELU}(W_1 \mathbf{h})$，`Linear(emsize → nhid) → GELU → Linear(nhid → n_out)`
- **无位置编码**（tabular 数据行没有顺序语义，加位置编码反而引入错误归纳偏置）

### 其他架构细节

- 输出投影层（attention out_proj + FFN linear2）**零初始化**，训练初期等价于恒等映射，梯度更稳定
- v1 使用 `TransformerEncoderDiffInit`：每层独立随机初始化，而非共享同一初始化

---

## 三、训练数据生成

### v0

两种 prior，每次用其中一种生成 (x, y)：

- **GP prior**：x ~ Uniform[0,1]，y = GP(x)（RBF kernel，固定超参）。相似性驱动，适合时序/空间数据
- **BNN prior**：x ~ N(0,1)，y = 随机 MLP(x) + noise。层次化非线性函数，适合 tabular 数据

### v1 新增

**SCM（结构因果模型）prior**——v1 最核心的数据生成创新：

```
BNN（v0）：  x（输入） → MLP → y（输出）    x 是原因，y 是结果

SCM（v1）：  隐变量 causes → 大 MLP → 所有中间节点
                                          ↙           ↘
                              随机挑几个 → x    随机挑几个 → y
```

x 和 y 都是隐变量的下游效应，不是直接的 x→y 关系。这更符合真实 tabular 数据的结构（例如"年龄"和"收入"都受"工作年限"影响，而非前者直接导致后者）。

**两层采样结构**：

```
第一层（differentiable_prior.py）
    采样 MLP 结构超参（num_layers、hidden_dim、activation、noise_std 等）
    使用 meta 分布：分布的参数本身也是随机的（meta_trunc_norm、meta_gamma 等）
        ↓
第二层（mlp.py）
    用这套超参搭随机 MLP，生成 (x, y)
```

**Prior bag**：GP 和 BNN/SCM 按权重随机混合，实际训练中权重极度偏向 BNN（`prior_bag_exp_weights_1 ∈ [1M, 1M+1]`，GP 几乎不被选中）。

### get_batch 返回三个值

```python
return x, y, y   # (x, y_noisy, y_clean)
```

第三个本应是无噪声的 y_clean，但两个版本都没有单独追踪，直接用同一个 y 填充。

---

## 四、任务类型与输出

### v0

- **回归（主推）**：Riemann Distribution（BarDistribution）
  - 连续 y 离散化成 100 个 bin，输出每个 bin 的概率——把回归变成分类
  - Bin 边界训练前从 prior 采样确定，**固定不变**，按等概率划分
  - 推理时提取预测值：`均值 = Σ(bin中心 × 该bin概率)`
- **二分类**：把连续 y 二值化（> median → 1），输出 2 维 logits

### v1

- **多分类（最多 10 类）**：
  - `MulticlassRank`：用随机边界把连续 y 切成 2–10 个整数类别
  - Decoder 固定输出 **10 维 logits**（`max_num_classes=10`）
  - 推理时只取前 `num_classes` 列做 softmax
  - 超过 10 类直接报错

训练时 `num_classes` 每个 batch 随机采样（50% 概率是二分类，50% 均匀采样 2–10）。

---

## 五、训练配置对比

| 配置项 | v0 | v1（实际训练） |
|--------|----|----|
| emsize | 512 | 512 |
| nlayers | 6 | 12 |
| nhid | 1024 | 1024 |
| 参数量 | ~13M | ~26M |
| batch_size（有效） | 1000 | 64（8×梯度累积8） |
| lr | 0.001 | 0.0001 |
| bptt（序列长度） | 10 | 1024 |
| epochs × steps | 200×10 = 2000步 | 400×1024 = 409,600步 |
| 合成 datasets 总量 | 200万 | 2600万 |
| Mixed precision | 无 | GradScaler + autocast |
| 分布式训练 | 无 | DistributedDataParallel |
| 训练时长 | 未提及 | 单张 V100，约 1 天 |

v1 参数量是 v0 两倍，训练计算量大几个数量级（序列长度增加 100 倍，步数增加 200 倍）。

---

## 六、Inference

### v0

直接一次前向传播输出预测，无预处理，无 ensembling。

### v1 新增

**预处理**（`preprocess_input`，只在推理时执行，用训练集 fit）：

```
z-score 归一化
    → 删除常数列
    → PowerTransformer（Yeo-Johnson）
    → remove_outliers
    → normalize_by_used_features
    → 补零到 max_features=100
```

**Ensembling**（默认 3 次，`N_ensemble_configurations=3`）：

三个维度的笛卡尔积，取前 N 个配置：

| 维度 | 含义 |
|------|------|
| preprocess_transform | `['none', 'power_all']` 两种预处理 |
| feature_shift | 特征列循环偏移（消除列顺序偏置） |
| class_shift | 类别标签循环偏移（消除类别顺序偏置） |

本质是一个 **hack**：模型在架构上对列顺序和类别顺序有隐含偏置，通过多次 shift 取平均来近似消除，而非从架构上根本解决。代价是推理时间增加 N 倍。v2 通过随机列向量编码从架构层面解决了这个问题。

---

## 七、V0 vs V1 核心差异

| 方面 | v0 | v1 |
|------|----|----|
| **输出 head** | 100-bin Riemann（回归）+ 2 维（二分类） | 固定 10 维 logits（多分类，≤10 类） |
| **Prior 种类** | GP + BNN | GP + BNN + SCM |
| **超参采样** | 固定超参 | 两层采样：meta 分布采结构超参 → 随机 MLP 生成数据 |
| **Prior 混合** | 无 | Prior bag（权重极度偏向 BNN/SCM） |

v1 的核心架构与 v0 几乎一致，主要差别集中在数据生成的多样性和输出 head 的任务类型上。
