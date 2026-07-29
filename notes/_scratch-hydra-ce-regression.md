# scratch — Hydra §4:CE~DE 回归、R²=variance explained、92% vs 70%

> 临时记录。对象:McGrath et al. *Hydra Effect* (2307.15771) §4 量化。
> 交叉引用:llm-self-repair/hydra-tech-core.md §4 · [[_scratch-tabular-logit-lens-vs-hydra]]

---

## 1. 92% 具体怎么算(CE~DE 回归)

固定一个**消融层** m(如 m=23),**跨 1209 个 counterfact prompt**,每个 prompt 出一对数:

- 对每个 prompt $i$:
  - 干净跑 → 各层直接效应 $\Delta_{\text{unembed},l}$(蓝)。
  - resample-ablate 第 m 层 → 重算下游各层 $\tilde\Delta^{m}_{\text{unembed},l}$(红)。
  - $x_i = DE = \Delta_{\text{unembed},m}$(被消融层自己那份直接效应)。
  - $y_i = CE = \sum_{\text{下游}l}(\tilde\Delta^{m}_{\text{unembed},l} - \Delta_{\text{unembed},l})$(下游补回来多少;红−蓝)。
- 得 ~1209 个散点 $(x_i,y_i)$ → 拟合直线 $CE\approx a+b\cdot DE$ → 斜率 $b\approx0.7$,$R^2=0.92$。
- **每个消融层各拟合一次**,R² 随层变;**layer 23 的 R² 最高(=92%)= "最佳层"**。

---

## 2. 线性回归 + R² 复习

拟合 $\hat y_i=a+bx_i$,最小化残差平方和 $\text{SS}_{\text{res}}=\sum_i(y_i-\hat y_i)^2$。

三个方差量:

- **总变异** $\text{SS}_{\text{tot}}=\sum_i(y_i-\bar y)^2$ —— CE 本身绕均值散多开。
- **残差变异** $\text{SS}_{\text{res}}$ —— 用直线预测后还剩多少散不掉。
- **被解释变异** $=\text{SS}_{\text{tot}}-\text{SS}_{\text{res}}$ —— 直线吃掉的那部分。

$$\boxed{R^2=1-\frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}=\frac{\text{被解释变异}}{\text{总变异}}}$$

- 字面:**CE 的总变异里,多大比例能被 DE 通过这条直线解释掉**。
- $R^2=0.92$ = CE 跨 prompt 的变化 **92% 由 DE 决定**,剩 8% 是绕线噪声。
- 原文 "variance in $\Delta_{\text{ablate}}$ explained by $\Delta_{\text{unembed}}$" 就是这个东西——**和 linear regression 是同一件事**,不是另一个指标。

---

## 3. 和相关系数的关系(解之前的疑问)

- **单变量线性回归**:$R^2=r^2$,$r$ = **Pearson 相关系数**。
- $R^2=0.92 \Leftrightarrow r\approx0.96$,高度线性相关。
- **是 Pearson(线性),不是 Spearman(秩)** —— 已确认原文是线性回归。

---

## 4. 别混:92% 和 70% 是两个不同的东西

同一条回归线,两个数量的是不同性质:

| 数 | 来自 | 量的是 |
|---|---|---|
| **斜率 $b\approx0.7$** | 线的陡峭度 | 补偿**有多少**(补回 70%,不完全)= 量级/完整度 |
| **$R^2=0.92$** | 点贴线有多紧 | 补偿**有多稳/成比例**(跨 prompt 一致)= 系统性 |

- 两者**独立**:可斜率 0.7 但 R² 低(平均补 70% 但每 prompt 乱跳);也可 R² 高(几乎每次精确 ~70%)。
- 原文两个都有:补偿既**实质**(70%)又**一致**(92% 贴线)。
- **R² 高才是"补偿是系统性反应、非巧合"的证据**;若补偿随机,CE 不跟 DE 走,R² 趴 0 附近。

---

## 5. 关联:CE 与 IE 的符号(接 §3)

- "层存在"约定下:$DE=+\Delta_{\text{unembed},m}$,$TE=DE+IE$,自修复 ⟺ IE 负。
- 三行推导:无间接效应时敲掉该掉 DE;实际掉 TE;差额 = 下游补偿 CE → $TE=DE-CE$;配 $TE=DE+IE$ 得 **$IE=-CE$**。
- ⚠️ 换成"消融动作"约定(`do(A^m=ã^m)`)则 $IE=+CE$,符号翻一次,物理一样(下游补了 CE)。
