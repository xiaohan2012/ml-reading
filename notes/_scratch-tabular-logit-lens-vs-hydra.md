# scratch — tabular logit lens 的 fine-tune 到底测的是什么 & 和 Hydra self-repair 的语义差

> 临时讨论记录(未定稿)。对象:Balef et al. *Is One Layer Enough?* (ICML 2026) 的 Exp 4 / Exp 6,
> 与 McGrath et al. *Hydra Effect* (2023) 的 direct/total effect 框架对照。
> 交叉引用:[[balef2026onelayer-experiments]] · [[exp4-code-walkthrough]] · llm-self-repair/hydra-tech-core.md

---

## 0. 起点:一个担忧

- tabular logit lens = **每层单独 fine-tune 的 decoder** $g_\ell$(从原始 decoder 初始化续训,数据 = TabICL 合成 prior)。核对无误(Exp4 note :178)。
- 标准 logit lens **不 fine-tune**,其读数 = 那一层对最终 logit 的**直接贡献** = Hydra 的 $\Delta_{\text{unembed}}$。
- 担忧:一旦 fine-tune,"direct-effect 读法"被破坏,丢掉可解释性红利。
  - 那 fine-tune 图什么?
  - 更进一步:就算在 fine-tuned lens 下看到 self-repair,它和 Hydra 里的 self-repair **也不是同一个意思**(因为 Hydra 没 fine-tune)。

**结论先行:担忧基本成立,但要拆成两半 —— 只有一半被 fine-tune 污染。**

---

## 1. 为什么 LLM 的 logit lens = direct effect

三件事同时满足才成立:

1. 残差流**可加**;
2. 各层写进**同一个 basis**;
3. 末端是**固定线性** $W_U$。

于是 "把 $W_U$ 套到中间层输出" = 那层对最终 logit 的**直接贡献** = $\Delta_{\text{unembed}}$。
**readout 共享且线性,是 direct-effect 语义的前提。**

---

## 2. TFM 破坏了这个前提 → 所以要 fine-tune

- TFM 的 decoder **不是**共享 basis 的固定线性 unembed,且与**最后一层**表示空间紧耦合;中间层不在它的输入流形上。
- 原始 decoder 直接套中间层 → 熵极高、极不自信、信号失真(论文明说,Exp4 note :174)。
- 这就是 LLM 圈 **Logit Lens vs Tuned Lens**(Belrose 2023)的老矛盾。**tabular logit lens = Tuned Lens 的表格版。**

**fine-tune 买到 / 赔掉:**

| | 内容 |
|---|---|
| 买到 | **可读性**。没有它,中间层读数是噪声,连"自我修复"这个问题都问不出来。 |
| 赔掉 | 读数不再是"这层自己的直接贡献",而是"一个**训练过的探针**能从这层线性抽出多少"。能解码出**模型还没在用**的信息。direct-effect 语义没了。 |

**关键分辨:fine-tune 的意义取决于问的是什么问题。**

- 论文自己的论点 = **冗余 / 信息早已形成**("同一件事反复精炼,深度冗余")→ 这是**可解码性**命题,不是因果机制命题。
  → 对它,tuned lens 恰恰是**对的工具**,direct-effect 纯度**无关紧要**(他们要的就是"信息第几层已线性可读")。
- 语义冲突只在 **想给 Exp 6 叠加 Hydra 式机制含义** 时才出现。

---

## 3. Exp 6 的 self-repair ≠ Hydra 的 self-repair —— 切成两半

**Hydra 的 self-repair 指纹 = 两个量脱钩:**

- $\Delta_{\text{unembed}}$(需**未调** lens,= 直接效应)
- $\Delta_{\text{ablate}}$(**根本不需要 lens**,直接量最终输出 loss,= 总效应)

指纹 = 这俩不一致。**总效应那一半从不依赖 lens。**

**Exp 6 做法**(note :246–256):skip 第 $m$ 层 → 用 tuned lens 读 $\ell>m$ 每层 AUC → 看末层是否恢复到 baseline。拆开:

| 部分 | 用什么测 | 和 Hydra 的关系 |
|---|---|---|
| **① 终点**:"skip 中层 → 最终 AUC 几乎不掉、后面爬回来"里的**最终层**恢复 | 真 decoder / 真输出 | **同义**于 Hydra 的 $\Delta_{\text{ablate}}$ 补偿。**干净,和 fine-tune 无关。** |
| **② 机制**:"某下游层**增加了自己的直接贡献**来补偿" | Hydra 用 $\Delta_{\text{unembed}}$;tabular 用 **tuned lens** | **语义岔了。测不忠实。** |

**② 为什么测不忠实:** tuned lens 下,ablate 后某层 AUC 上升,可能是

- (a) 该层真的多写了预测信号(真 Hydra 补偿),**也可能是**
- (b) 那信息**一直**潜伏在该层,只是变得更线性可解码,模型计算根本没变 —— 探针 $g_\ell$ 分不清"模型修复了"和"信息本来就在那、我这个训练过的探针把它捞出来了"。

> **一句话(可入正式笔记):** self-repair 作为"最终输出恢复"和 Hydra 同义、不需未调 lens;
> self-repair 作为"逐层直接贡献上升"用 tuned lens 测不忠实,所以 Exp 6 的恢复轨迹是比 Hydra
> "$\Delta_{\text{unembed}}$ vs $\Delta_{\text{ablate}}$ 脱钩"**更弱、语义不同**的构造。

---

## 4. "未 fine-tune 的 direct-effect 读法其实还在图里" —— 在哪儿

- **图**:Figure 6 = tabular logit lens / early-exit 图(仓库图 `notes/figures/balef2026/exp4_logit_lens.png`)。
- **脚本**:`Experiments/plots/01_early_exit.py`(已核实)。
  - `probings = ['decoder', 'finetuned_decoder']`(:41)
  - `colors = ["#0072B2", "#D55E00"]`(:74)
  - `display_name = "Original decoder" if probing=="decoder" else "Individual decoder"`(:132)
- **映射(核实过):**
  - 🔵 **蓝线 `#0072B2` = "Original decoder"** = 原始**未 fine-tune** decoder 逐层套用 = **direct-effect 读法 baseline**(中间层下陷 / 一路落后,TabPFN v2 最明显)。
  - 🟠 **橙线 `#D55E00` = "Individual decoder"** = 每层 fine-tune。
  - ⬛ 黑虚线 = "Full Model"(完整模型最终性能参考);灰细线 = 各数据集单独轨迹。
- 所以 direct-effect 读法**没被完全丢**,它就是那条蓝线,只是给弱/不校准读数。论文用"蓝落后于橙"来论证:后期层在做"表示对齐 decoder basis",而非"生成新预测信息"(Exp4 note :195)。

---

## 5. 对迁移项目的具体后果

- Hydra 的**整个**指纹依赖"一个直接效应量 vs 一个总效应量"的**分歧**。
- 若 TFM 里唯一的逐层读数是 tuned lens → **失去了构造干净 $\Delta_{\text{unembed}}$ 的能力**。
- 要忠实移植 Hydra,需要一个**不依赖训练探针**的逐层直接效应量,例如:
  1. 把每层的**残差增量**过**未调的最终 decoder**(即蓝线那套,哪怕低置信)—— 保留 direct-effect 语义;或
  2. 干脆在真输出上做 path-patching / 直接效应 ablation(Hydra $\Delta_{\text{ablate}}$ 那套)。
- **tuned-lens 的 AUC 轨迹不是这个东西。**
- 相关但更浅的担忧:论文 A9 承认 finetuned decoder(TabICL 合成数据训)对真实数据泛化未验证 —— 那是**分布**担忧;本文这条是**"测的到底是什么量"的语义**担忧,更根本。

---

## 6. Tuned Lens 原文佐证(Belrose et al. 2023, arXiv 2303.08112)

> 源码 `/tmp/tunedlens_src/arxiv.tex`。标题 *Eliciting Latent Predictions from Transformers with the Tuned Lens*。

**motivation:logit lens 三个毛病**(§2–3)

- **(a) 脆**:GPT-Neo/BLOOM/OPT 直接失效;GPT-Neo-2.7B 第 21 层前读不出;BLOOM/OPT 过半层 top-1 = **输入 token 本身**。
- **(b) representational drift**:hidden state 协方差随深度漂开,末层突变 → 末层 $W_U$ 误读早层;外加 rogue dimensions。
- **(c) bias**:即便能用也是**有偏**估计(GPT-Neo-2.7B 4–5 bits KL),破坏"信念随证据更新"解释(可 Dutch-book 套利)。

**修复**:每层学一个 affine translator $(A_\ell,b_\ell)$,再套**原** $W_U$:
$$\mathrm{TunedLens}_\ell(h_\ell)=\mathrm{LN}[A_\ell h_\ell+b_\ell]W_U$$
- $A_\ell$ 纠漂移(换基),$b_\ell$ 去偏。**蒸馏损失**(KL 到末层 logits,不是 ground-truth)→ 探针学不到模型没有的额外信息。
- 不给每层学新 unembedding(区别 Alain & Bengio 2016)→ 矩阵 $d\times d$ 而非 $|V|\times d$。

### 6.1 关键:tuned lens = belief tracker,不是 contribution meter

| | Hydra 的 logit lens | Tuned lens |
|---|---|---|
| 读出 | 这层**自己写了什么**(直接投影)= direct effect $\Delta_{\text{unembed}}$ | 模型此刻对**最终答案**的隐含信念(蒸馏到末层)= belief / prediction trajectory |
| 参数 | 零 | 学 $(A_\ell,b_\ell)$ |
| 代价 | 脆、漂移、有偏 | **丢掉 direct-effect 语义**,换可读 / 低 ppl / 无偏 |

- **tuned lens 设计上就放弃"直接贡献"读法**,去换一条可信的信念轨迹 → 这是我们担忧的**数学根源**,不是 tabular 独有的取舍。
- LLM/TFM 差别:LLM tuned lens = affine translator + 固定线性 $W_U$,**整体仍线性**(只换基);tabular 那篇是**重训非线性 MLP decoder**,比 tuned lens **更远**离直接效应。→ 印证 §2 的"第二条腿(非线性 head)"是 **tabular 特有**的额外偏离,LLM tuned lens 里不存在。

### 6.2 CBE 因果保真度验证 = tabular 那篇欠的作业

- §4 *Measuring Causal Fidelity*:Belrose 他们**自己就担心** learned probe 会依赖**与模型性能无关的 spurious 特征**(= 我们对 tabular Exp 6 的原话)。
- 于是做**因果验证**:**CBE**(causal basis extraction)找出 tuned lens 依赖的方向 → 去模型里 ablate 这些方向 → 证明"对 lens 重要的方向对模型输出也重要"(Pythia-410M,Spearman ρ=0.89),且散点图**右下角空**(没有"对 lens 重要、对模型不重要"的点)。
- 另有 **stimulus-response alignment**:扰动后 lens 输出的移动方向 vs 模型输出的移动方向对齐(Aitchison 几何),tuned lens 比 logit lens 更对齐。
- **tabular 那篇对它的 fine-tuned decoder 完全没做这类验证**——连"tabular logit lens 忠实于模型"都没证,更别提拿它推 self-repair 机制。**→ 这是可直接补的实验:对某层 fine-tuned decoder 做 CBE + 模型侧 ablate,验证方向一致性。**

---

## TODO / 待决

- [ ] 决定这段落定后并入 `balef2026onelayer-experiments.md`(Exp4/Exp6)还是 `hydra-tech-core.md` 交叉引用。
- [ ] 方案 1(蓝线残差增量法)是否够灵敏做出 Δ_unembed–Δ_ablate 脱钩图?小实验验证。
- [ ] 对 tabular fine-tuned decoder 补 CBE + 模型侧 ablate 的因果保真度验证(§6.2),补论文欠账。
