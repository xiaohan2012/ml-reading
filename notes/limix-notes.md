# LimiX 阅读笔记

**名称：** LimiX: Unleashing Structured-Data Modeling Capability for Generalist Intelligence
**arXiv：** 2509.03505
**会议：** COLM 2024
**团队：** 多机构（38位作者）
**LaTeX：** `wiki/raw/2509.03505/`

---

## 一、定位

LimiX 的出发点和 TabICL/TabFM 根本不同：

- TabICL/TabFM：学 $P(Y|X)$，推理时做 ICL，只做分类/回归
- LimiX：学**联合分布** $P(X, Y)$，一个模型同时支持分类、回归、缺失值填补、数据生成、OOD 预测

这个选择决定了它整个训练范式——用**掩码建模**而不是直接 label 预测。

---

## 二、架构

### 单元嵌入
- 每个 cell $(i,j)$ 用**2层 MLP + LayerNorm** 投影到 $\mathbb{R}^p$（$X$ 和 $Y$ 各自独立的 embedding 模块）

### 列身份编码（DFE）
- 问题：attention score 只看 cell value，不知道来自哪列
- 解法：Discriminative Feature Encoding（DFE）——低秩列标识符
  - 每列 $j$ 有低维码 $u_j \in \mathbb{R}^s$（$s = p/4$），初始化近似正交
  - 线性映射 $E \in \mathbb{R}^{s \times p}$ 升维：$e_j = u_j E$
  - 加到 cell embedding：$\tilde{x}_{i,j} = x_{i,j} + e_j$（类似 positional encoding，但作用在列轴）

### Transformer 块结构
- 共 **12 块**，每块：**2次 feature-axis attention + 1次 sample-axis attention**（不对称 2:1）
  - 消融实验显示：feature attention 太少会欠拟合异构 schema，增加 feature pass 收益明显
- 两个规模：LimiX-16M 和 LimiX-2M

---

## 三、训练：CCMM

**Context-Conditional Masked Modeling（上下文条件掩码建模）**

每个 episode：
1. 把数据集的行分成**上下文集**（context）和**查询集**（query）
2. 随机掩盖查询集里的一部分 cell
3. 模型在看到上下文全部 + 查询集可见 cell 的条件下，预测被掩盖的 cell

和 BERT 的区别：上下文是推理时可以替换的非参数记忆（数据集级别的自适应），不需要 finetune。

掩码策略多样：逐 cell 掩码 + 逐列掩码 + block 掩码，掩码率 [0.1, 0.4]。引入 mask density token（标量编码当前掩码比例），减少预训练-推理 mismatch。

---

## 四、合成数据生成

- 用**层次化 SCM + DAG** 生成：先建 DAG，每条边用 MLP / 卷积层 / 决策树定义函数
- 两种采样策略：
  - **Graph-aware sampling**：约束采样空间，保证数据满足图结构
  - **Solvability-aware sampling**：高/中/低可解性按比例混合，提升泛化

---

## 五、推理：Attention-Guided Retrieval

推理时做 **2-pass + 多管道集成**：
- 第一遍：用全量 context 做前向，提取 feature-level attention $a_f$ 和 sample-level cross-attention $a_s$
- 合并：$a_{sf} = a_s \cdot a_f$（用 feature 重要性加权 sample 相似度）
- 第二遍：用检索到的高分 context 样本再做一次前向

集成规模：分类 4 路、回归 8 路（每路随机列排列 + 特征变换）

---

## 六、和 TabICL/TabFM 的主要区别

TabICL 和 TabFM 属于同一范式（条件预测 ICL），LimiX 是截然不同的另一类方法。

| 设计点 | ICL 类（TabICL / TabFM） | LimiX-16M |
|--------|--------------------------|-----------|
| 训练目标 | $P(Y\|X)$，只预测 label | **$P(X,Y)$ 联合分布，掩码重建任意 cell** |
| 架构组织 | 分段管道：Col→Row→（ICL） | **单一 12 块 Transformer，特征轴×2 + 样本轴×1** |
| 列身份编码 | TAE / Fourier features | **DFE（低秩正交列码，类比列方向 PE）** |
| 支持任务 | 分类 + 回归 | **分类 + 回归 + 缺失值填补 + 数据生成 + OOD** |
| 推理集成 | 随机列排列集成 | **attention 引导样本检索 + 多路集成** |
| 合成 prior | SCM 类（TabICL 详细，TabFM 黑盒） | **层次化 SCM + DAG，公开，含可解性控制** |

---

## 七、实验结果

- **分类**：在 BCCO-CLS / OpenML-CC18 / TALENT-CLS / TabArena / TabZilla / PFN-CLS 六个 benchmark 上，LimiX-16M 全面第一，**唯一能稳定超过 AutoGluon 的 ICL 模型**
- **回归**：同样各 benchmark 第一
- **缺失值填补**：零 finetune，优于 KNN / MICE / MissForest / HyperImpute 等需要训练的方法
- **数据生成**：优于 TabPFN-v2（后者只建模 $P(Y|X)$，无法生成 $X$）
- **OOD（TableShift）**：ID AUC 0.848 / OOD AUC 0.806，OOD Rank 第一（1.3）

**注意：** 自评，用了自建 BCCO benchmark，不同团队评测结论未必一致。
