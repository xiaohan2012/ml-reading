# TabICL v2 阅读笔记

**论文：** TabICLv2: A better, faster, scalable, and open tabular foundation model (qu2026tabiclv2, ICML 2026)
**arXiv：** 2602.11139
**代码：** `soda-inria/tabicl` tag `v2.1.1`
**LaTeX：** `wiki/raw/2602.11139/main_icml.tex`

---

## 一、三个核心贡献

论文摘要明确的三个支柱，按消融实验影响力排序：

1. **新的合成数据生成引擎**——更高预训练多样性（影响最大）
2. **架构创新**——核心是 QASSMax + TAE（Target-Aware Embedding）；QASSMax 让模型不需要在超长序列上预训练就能泛化到大规模数据，TAE 在 TF_col 阶段就注入标签信息；两者贡献相当（各约 100 Elo）
3. **优化预训练流程**——核心是换用 Muon 优化器

整体架构流水线与 V1 相同（TF_col → TF_row → TF_icl），复杂度仍为 $O(n^2 + nm^2)$，在此基础上叠加多个改进。

---

## 二、数据生成（Prior）`【核心】`

### V1 vs V2 的根本区别

消融实验发现：用 V1 prior 训练 V2 架构，性能不升反降；用 V2 prior 训练 V1 架构，也只能打平 V1。两者缺一不可——**prior 的多样性是整个系统的瓶颈**。

### 1. 图结构：链式 → 任意 DAG `【核心】`

- **V1**：MLP 堆叠，本质是链状结构
- **V2**：先采样随机 DAG（节点数 $\sim \text{LogInt}(2, 32)$），再在上面跑数据生成

建边用 **random Cauchy graph**：对所有节点对 $(i, j)$，

$$p_{ij} = \text{sigmoid}(A + B_i + C_j)$$

- $A$：全局连接度（Cauchy 随机变量）
- $B_i$：节点 $i$ 的出度倾向
- $C_j$：节点 $j$ 的入度倾向

选 Cauchy 而不是 Gaussian 的原因：Cauchy 是**重尾分布**，偶尔产生极大值，使得某些节点连接极多、某些极少——生成的图结构比均匀 Bernoulli 更多样。

### 2. 节点函数：2 类 → 8 类 `【核心】`

V1 只有 MLP 层和树模型（XGBoost/RF）。V2 扩展到 8 种随机函数：

| 函数 | V1 有 | 说明 |
|---|---|---|
| RandomNNFunction（MLP） | ✓ | 随机 MLP，1-3 层，随机宽度和激活 |
| RandomTreeFunction | ✓ | **对称树（CatBoost 风格）**，V1 用 XGBoost/RF |
| RandomDiscretizationFunction | - | 映射到最近邻中心，生成分段常数函数 |
| RandomGPFunction | 部分 | V1 用 GP 随机激活，V2 扩展为完整多元 GP，核函数随机采样（幂律尾控制平滑度） |
| RandomLinearFunction | - | 线性变换，决策边界为超平面 |
| RandomQuadraticFunction | - | 二次函数，椭圆形决策边界 |
| RandomEMFunction | - | EM 簇分配逻辑，生成带"平台"的分段函数 |
| RandomProductFunction | - | 组合其他函数的乘积 |

多父节点时支持两种聚合方式：
- 所有父节点拼接后统一跑一个函数
- 各父节点分别跑函数，再用 sum / product / max / logsumexp 聚合

### 3. 数据过滤（V1 没有）`【次要】`

生成后用 **ExtraTrees** 检验数据集质量，过滤掉约 35% 分类、25% 回归数据集。

**ExtraTrees** 是比 RandomForest 更快的树集成方法——RandomForest 找最优分裂点，ExtraTrees 用随机分裂点，速度更快。

过滤逻辑：

- 训练一个轻量 ExtraTrees（25 棵树，最大深度 6，bootstrap=True）
- 用 out-of-bag 预测做 200 次 bootstrap 检验
- 如果 95% 以上的 bootstrap 样本里，ExtraTrees 都没比"预测均值"更好，则丢弃该数据集

**为什么对合成数据用弱分类器来过滤是合理的**：这里过滤的是合成数据，不是真实数据。随机图 + 随机函数的组合大概率会产生"无信号"数据集（$x$ 和 $y$ 的节点在 DAG 里没有共同祖先，或者随机函数叠太多层后信号被淹没）。这种情况下 ExtraTrees 学不到，说明不是模型太弱，而是**数据本身就没有有效信号**——拿来训练只会让模型学到"随便预测都行"。

V2 也在图层面做早期过滤：如果 $x$ 和 $y$ 对应节点在 DAG 里没有共同祖先，直接重新采样图，不进入数据生成阶段。

---

## 三、架构创新

### 3.1 QASSMax（Query-Aware Scalable Softmax）`【核心】`

#### Notation

Multi-head attention 里 query 按 head 切分。以 $d=512$，8 个 head 为例，每个 head 维度 $d_\text{head}=64$：

- $h \in \{1,\ldots,8\}$：第几个 attention head
- $q_h \in \mathbb{R}^{64}$：第 $h$ 个 head 的 query 向量
- $q_{hi}$：$q_h$ 的第 $i$ 个元素（标量）

#### 问题：Attention Fading

标准 softmax attention：

$$\text{score}(q_h, k_h^{(j)}) = \frac{q_h^\top k_h^{(j)}}{\sqrt{d_\text{head}}}, \qquad \text{attn}_h = \text{softmax}\Big(\big[\text{score}_1, \ldots, \text{score}_n\big]\Big)$$

分母 $\sum_{j=1}^n \exp(\text{score}_j)$ 随 $n$ 线性增长，导致 softmax 输出趋向均等 $1/n$——$n$ 越大，模型越难聚焦到相关 token。对在 1K 行上预训练、推理时遇到 50K 行的 TabICL 影响尤其大。

#### 已有方案：SSMax

先对 query 缩放，再算 attention：
**Mitra**
$$\tilde{q}_{hi} = q_{hi} \cdot s_h \log n$$

$$\text{score}_j = \frac{\tilde{q}_h^\top k_h^{(j)}}{\sqrt{d_\text{head}}}$$

$s_h$ 是每个 head 一个可学习标量，$\log n$ 是常量（序列长度的对数，不可学习），两者相乘才是最终缩放系数。

**为什么是 $\log n$**：当所有 score 大致相等时，softmax 分母约为 $n \cdot \exp(\text{score})$，取 log 后分母贡献恰好是 $\log n$——即 softmax 的"有效温度"随 $n$ 增长的速率正好是 $\log n$。用 $\log n$ 缩放 query 等价于把所有 score 乘以 $s_h \log n$，正好抵消这个增长，使 attention 分布尖锐程度与 $n$ 无关。论文引用的理论工作（critical-attention-scaling）也证明 $\log n$ 缩放是维持 attention 尖锐度的**必要条件**，不是随便选的。

缺点：整个 query 向量乘同一个标量 $s_h \log n$（逐元素写法只是为了和 QASSMax 的 notation 对齐），每个 head 只有一个自由度，表达能力有限。

#### V2 的改进：QASSMax

把标量 $s_h$ 换成两个 MLP 的乘积，**每个维度 $i$ 独立缩放**：

$$\tilde{q}_{hi} = q_{hi} \cdot \underbrace{\text{MLP}_\text{base}(\log n)_{hi}}_{\text{base scaling}} \cdot \underbrace{\big(1 + \tanh(\text{MLP}_\text{gate}(q_h)_i)\big)}_{\text{query-aware gating}}$$

$$\text{score}_j = \frac{\tilde{q}_h^\top k_h^{(j)}}{\sqrt{d_\text{head}}}$$

两个 MLP 的输入输出：

| MLP | 输入 | 输出 | 含义 |
|---|---|---|---|
| $\text{MLP}_\text{base}$ | $\log n$（标量） | $\mathbb{R}^{H \times d_\text{head}}$ | 随序列长度变化的基础缩放，所有 head 所有维度各一个系数 |
| $\text{MLP}_\text{gate}$ | $q_h \in \mathbb{R}^{d_\text{head}}$ | $\mathbb{R}^{d_\text{head}}$ | 随 query 内容变化的门控，每个 head 独立运行 |

- $\tanh$ 把 gating 限制在 $(0, 2)$，不会盖掉 $\log n$ 的主导作用
- 两个 MLP 都是 2 层、64 个隐藏神经元、GELU 激活

**应用位置**：TF_col 的 MAB₁（inducing points 聚合那步）+ TF_icl 全程。

**验证实验**："大海捞针"分类任务——训练集里只有 1 个 anchor 样本有效，负样本逐渐增加到 15K。无 SSMax 时准确率崩溃；QASSMax 全程保持 100% 准确率。

代码：`src/tabicl/_model/ssmax.py`

### 3.2 Target-Aware Embedding（TAE）`【核心】`

**V1**：标签只在 TF_icl 阶段注入（`y_encoder` 加到训练行 embedding 上）。

**V2**：在 TF_col 阶段就注入，对训练行的每个 feature token 都加上标签 embedding：

$$E_2[i,j] = E_1[i,j] + \text{Embed}_\text{TAE}(y_i), \quad i \in \mathcal{D}_\text{train}$$

- 分类：可学习的 lookup table
- 回归：线性层

好处：
- 让 TF_col 在 embed 时就能感知标签，两个分布相同的列如果和 $y$ 的关联不同，TF_col 就能区分它们——进一步缓解表示坍塌
- 信息更早流入模型，消融实验显示收益约 100 Elo

**多分类处理（>10 类）**：V2 引入 mixed-radix ensembling——把标签分解为 $D$ 个数字（混合进制），每个数字跑一次 TF_col，结果平均。比 V1 纯 hierarchical classification 更系统。

### 3.3 Repeated Feature Grouping（解决表示坍塌）`【次要】`

**V1 的问题**：每列单独 embed，当多列分布相同时 TF_col 输出几乎一样，TF_row 无法区分列身份（表示坍塌）。V1 用 RoPE 缓解。

**V2 的解法**：把每 3 列编码为一组，每列参与多个不同的组，不同组内邻居不同，自然打破对称性：

$$E_1[i,j] = \text{Lin}\big(x_{i,j},\ x_{i,(j+1)\bmod m},\ x_{i,(j+3)\bmod m}\big)$$

偏移量选 $(0, 1, 3)$ 是有设计的：对 $\geq 7$ 列的表，任意两列最多共同出现在一个组里，保证组间独立性。

- `col_affine=False`（默认）：TF_col 直接输出 embedding，不用 V1 的 FiLM 调制
- 每列仍然产生相同数量的 token（"same"模式），不损失特征粒度

**消融实验结论**：marginal 改进，不是 essential。论文原话（Section 6 Ablation）："Repeated feature grouping and prior filtering yield smaller gains"——相比 TAE、Muon、QASSMax 各贡献约 100 Elo（64% win rate），feature grouping 的收益明显更小。

### 3.4 回归支持（新增）`【新功能】`

V2 新增回归任务，预测 999 个分位数（概率水平 $0.001, 0.002, \ldots, 0.999$），用 pinball loss 训练。

推理时：
- 点估计：直接对 999 个分位数取平均
- 概率预测：构造完整分布（monotonicity via sorting，尾部用参数化指数模型外推）

代码：`src/tabicl/_model/quantile_dist.py`

---

## 四、预训练流程

### Muon 优化器 `【核心】`

**AdamW → Muon**：Muon 在 Nesterov momentum 之后对每层权重矩阵做正交化（Newton-Schulz 迭代），等效于把更新方向投影到矩阵谱范数球上。同等步数下收敛更快，允许更大学习率（V1 用 1e-4，V2 用 8e-4）。

配套：
- **Cautious weight decay**（参数 0.01）：只有"更新方向和参数同号"时才施加 weight decay，避免干扰有益梯度方向
- gradient clipping 从 1 提高到 10
- cosine schedule 贯穿三阶段

### 三阶段 Curriculum `【次要】`

| 阶段 | 步数 | 数据集规模 | 最大学习率 |
|---|---|---|---|
| Stage 1 | 500K | 1,024 样本，30-90% 训练 | 8e-4 |
| Stage 2 | 40K | 400-10,240 样本（log-uniform） | 1e-4 |
| Stage 3 | 10K | 400-60K 样本（log-uniform） | 2e-5 |

Batch size 从 V1 的 512 降到 64，步数更多但总数据集更少（35M vs V1 的 83M）。

**预训练成本**：24.5 H100-days（≈49 A100-days），低于 V1 的 60 A100-days，性能反而更高。

---

## 五、与 TabICL V1 的全面对比

| 方面 | TabICL V1 | TabICL V2 |
|---|---|---|
| **整体架构** | TF_col → TF_row → TF_icl | 相同，但各模块有改动 |
| **复杂度** | $O(n^2 + nm^2)$ | 相同 |
| **特征 embedding** | 每列单独 embed + FiLM 调制（`col_affine=True`） | 3 列一组循环分组（`col_affine=False`） |
| **表示坍塌** | RoPE 缓解 | Feature grouping + TAE |
| **标签注入时机** | 只在 TF_icl | TF_col 就注入（TAE） |
| **Softmax** | 标准 softmax | QASSMax（TF_col MAB₁ + TF_icl） |
| **多分类（>10）** | Hierarchical classification | Mixed-radix ensembling + hierarchical |
| **回归** | 不支持 | 999 分位数预测 |
| **Prior 图结构** | 链式 MLP/树 | 任意 DAG（random Cauchy graph） |
| **Prior 函数类型** | MLP + XGBoost/RF | 8 种（含 GP、Quadratic、EM 等） |
| **数据过滤** | 无 | ExtraTrees bootstrap 过滤（约过滤 35%） |
| **优化器** | AdamW | Muon + cautious weight decay |
| **学习率** | 1e-4 | 8e-4（Stage 1） |
| **预训练成本** | 60 A100-days | 49 A100-days |
| **任务** | 仅分类 | 分类 + 回归 |
