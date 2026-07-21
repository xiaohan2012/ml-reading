# TabFM 阅读笔记

**名称：** TabFM: A Zero-Shot Foundation Model for Tabular Data
**发布：** 2026年6月30日
**团队：** Google Research
**博客：** https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/
**权重：** HuggingFace `google/tabfm-1.0.0`（非商业许可）
**arXiv：** 无，只有博客文章

---

## 一、定位

和 TabICL 思路一致：用合成数据预训练，推理时做 in-context learning，零调参。

三段式架构（列注意力 → 行压缩 → ICL Transformer）和 TabICL 相同，但做了**两轮**列+行循环，TabICL 只做一轮。

---

## 二、和 TabICL 的主要区别

| 设计点 | TabICL V1 | TabICL V2 | TabFM |
|--------|-----------|-----------|-------|
| 单元嵌入 | 标量→Linear | 同左 | **Fourier features**（sin/cos，数值/类别各自频率组） |
| Col+Row 轮数 | 1轮 | 1轮 | **2轮** |
| Query 缩放 | 无 | QASSMax（log n 自适应） | PerDimScale（可学习，无 log n） |
| TAE | 无 | ✅ | ✅ |
| Feature grouping | 无 | ✅ size=3 | ✅ size=3 |
| 激活函数 | GELU | GELU | SwiGLU |
| ICL 层数 | 12 | 24 | 24 |
| 回归输出 | 无 | 999 分位数回归（完整分布） | 简单 RMSE（单值） |
| 训练 prior | 有详细说明 | 有详细说明 | **黑盒，不开源** |
| 论文 | ✅ | ✅ | ❌ 只有博客 |

---

## 三、实验结果

Benchmark：TabArena（Elo 积分，38 分类 + 13 回归数据集）

- TabFM（零调参）> AutoGluon 1.5（extreme，4小时）> TabPFN-3 > TabICLv2
- TabFM-Ensemble 再高一档

**注意：** Google 自评，TabICL 团队用不同 benchmark 时结论不一定一致。
