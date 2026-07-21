# TabPFN 2.5 阅读笔记

**报告：** TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models (Prior Labs Team, 2025-11-06)
**arxiv：** 2511.08667
**代码：** `PriorLabs/TabPFN` tag `v6.0.0` 起默认启用

---

## 一、核心定位

TabPFN 2.5 是在 v2 架构基础上的**规模化**版本，不是根本性的架构创新。核心设计不变：
- 交替 row/col attention
- 随机列身份（randomized attribute tokens）
- KV cache + multiquery

主要变化集中在：更深的网络、更大的 feature group、新的 thinking rows、更强的 prior。

---

## 二、架构变化

| 方面 | v2 | 2.5 |
|------|----|----|
| **层数** | 12 | 回归 18 层，分类 24 层 |
| **feature group size** | 2 | 3（每组 3 列合并成一个 token） |
| **x encoder** | `Linear(1 → d)` | 回归模型改为 2 层 MLP |
| **thinking rows** | 无 | 64 个可学习的额外行 |

**Thinking rows：**

在输入数据前面加 64 个可学习的行，灵感来自 LLM 的 thinking token：
- 给模型更多计算容量（额外的 attention 路径）
- 可以作为 attention sink，帮助模型忽略噪声行

---

## 三、数据生成

- Prior 更丰富、分布更多样，规模更大
- 额外发布 **Real-TabPFN-2.5**：在 43 个真实数据集（OpenML + Kaggle）上 fine-tune，性能进一步提升

---

## 四、超参搜索

用了自指策略：**用 TabPFN v2 本身作为 surrogate model** 来搜索 TabPFN 2.5 的超参——"TabPFN tunes TabPFN"。

约 50 个超参，先训练 ~100 个模型得到稀疏的超参-性能对，再用 v2 在 10,000 个配置上做插值预测，找到最优区域再做完整训练。

---

## 五、推理

- 比 v2 快 **1-2.3 倍**（更大的 feature group + FlashAttention-3 + 多 GPU 并行）
- 新增 **distillation engine**：把 TabPFN 2.5 蒸馏成 MLP 或 tree ensemble（TabPFN-2.5-as-MLP / as-TreeEns），用于低延迟部署场景

---

## 六、规模对比

| 模型 | 最大行数 | 最大特征数 | 层数 | 推理模式 |
|------|---------|-----------|------|---------|
| TabPFN v1 | 1,000 | 100 | 8 | ICL |
| TabPFN v2 | 10,000 | 500 | 12 | ICL |
| **TabPFN 2.5** | **50,000** | **2,000** | **18–24** | **ICL + MLP/Trees** |

---

## 七、性能

- 在 TabArena-lite 上超越所有单模型，包括调参后的 XGBoost、CatBoost
- 默认模式下对 XGBoost 胜率 100%（$\le 10,000$ 行、500 特征以内）
- 对 AutoGluon 1.4（4 小时调参集成）持平或接近

**许可证：** TABPFN-2.5 Non-Commercial License v1.0（训练代码未开源）
