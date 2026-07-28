# 论文一:Wang et al. 2022 — "Interpretability in the Wild: 一个 IOI 电路"

> arXiv 2211.00593 · Redwood Research · GPT-2 small 上的间接宾语识别(IOI)电路
> 导读定位:自修复系列的**起点**——本篇无意中记录下"敲掉主力头后有备份头补偿"和"负向头"两个反直觉现象,后三篇由此展开。

## 1. 这篇论文到底在做什么(全貌)

一句话:**一次"逆向工程"的示范——把 GPT-2 small 完成 IOI 任务的内部机制,拆解成一个由 26 个注意力头组成、人类可读的电路,并提出一套量化标准来检验"这个电路解释是否可信"。**

- 任务 IOI:句子如 "When Mary and John went to the store, John gave a drink to ___",模型要预测 "Mary"(没有重复出现的名字 = 间接宾语 IO)。
- 两个真正的重心:
  1. **能不能对一个"真实的"(in the wild)语言任务做端到端的机制解释**——不是玩具模型、也不是"大而化之"的高层描述。作者自称这是当时最大规模的一次自然行为逆向工程。
  2. **怎么判断一个电路解释是"对的"**——提出 legibility(可读性)+ faithfulness / completeness / minimality(忠实/完整/最小)四条标准,并用它们检验自己的电路,也坦白指出标准暴露的漏洞。
- 重心提醒:**这篇根本不是在研究自我修复**。自修复(Backup Name Mover Heads)是做"完整性检查"时**顺带撞见的一个意外现象**,只占一小节(见 §4)。

## 2. 理解这篇必须掌握的关键概念

后三篇共享一套 Elhage et al. 2021(“A Mathematical Framework”)的语言,在此讲透:

- **残差流(residual stream)**:贯穿始终的"信息总线"。输入嵌入是初值,之后每个注意力层、每个 MLP 都从它**读**、把结果**加**回去。关键性质:**线性可加**——第 L 层残差流 = 初始嵌入 + 之前所有 head 和 MLP 输出之和。这条线性假设是一切"归因"技术的地基。
- **把注意力头拆成 QK 和 OV 两个电路**:
  - $W_{QK} = W_Q W_K^\top$ 决定**注意力模式**(head 从哪个位置读)。
  - $W_{OV} = W_O W_V$ 决定**写什么内容**进残差流(读到的东西怎么被搬运)。
  - 让作者能分别问"这个头看哪里"和"这个头搬什么",是全篇分析的骨架。
- **电路(circuit)**:把模型看成计算图(节点 = head/MLP/嵌入,边 = 读写残差流的交互),电路就是负责某行为的**子图**。节点是"模型组件",比 Olah 2020 的"特征方向"版本更粗粒度。
- **Knockout / 消融(ablation)**:怎么"关掉"节点来测因果作用。
  - 不用**零消融**(设为 0,会把模型推到分布外,噪声大),而用**均值消融(mean ablation)**:用该 head 在参考分布上的平均激活替换它。
  - 参考分布 $p_{\text{ABC}}$:把句子里两个名字(IO, S)换成三个不相关随机名字,保留语法结构但抹掉任务信息。均值消融只删掉分布里**变化**的信息,保留**恒定**的信息(如"这个位置是个名字")。
- **Path patching(路径修补)——本篇提出的核心新技术**:
  - 比普通 activation patching 更精细:只沿**特定路径**(如 head $h \to \text{logits}$ 的直接路径,经残差和 MLP 但**不经其他 head**)注入来自 $p_{\text{ABC}}$ 的激活,测量 $h$ 对下游组件 $R$ 的**直接**因果效应(区分"直接" vs "被别的头中转的间接")。
  - 是"从 logits 往回一层层追信息流"的工具,后三篇都建立在它之上。
- **Logit difference(logit 差)**:任务度量 $=$ IO 的 logit 减 S 的 logit。用差反映"IO 概率高"且"IO 排在 S 前"。GPT-2 small 均值 logit 差 $3.56$。
- **Direct logit attribution / copy score**:把一个 head 输出投影到某词的 unembedding 方向 $W_U[\text{name}]$,看它"写了多少该词的 logit"。copy score $=$ 模拟该头完美注意某名字时该名字进 top-5 的比例(Name Mover 头 $>95\%$)。

## 3. 论证的主要结构

"**从输出端往回倒推 + 事后量化验证**"两段式:

**第一段:发现并解读电路**

人类算法:①找出所有出现过的名字 → ②划掉重复的 → ③输出剩下那个。从 logits 逆流而上,用 path patching 找头并逐类验证:

1. **Name Mover Heads($9.6, 9.9, 10.0$)**:直接影响 logits,注意 IO 并复制到输出(copy score $>95\%$)——对应第③步。
2. **Negative Name Mover Heads($10.7, 11.10$)**:反常——**朝名字的反方向写**,降低正确答案置信度(负 copy score $98\%$)。作者猜测是模型在"对冲"以避免过度自信时的高损失。**这是"自修复/copy suppression"故事的第一个种子。**
3. **S-Inhibition Heads($7.3, 7.9, 8.6, 8.10$)**:影响 Name Mover 的 **query**,抑制其注意重复的 S,使注意力偏向 IO。携带 token 信号(S 身份)+ position 信号(S1 位置),后者作用更大。
4. **Duplicate Token Heads**:在 S2 注意到 S1,标记"这个 token 重复了"。
5. **Induction Heads + Previous Token Heads**:用归纳机制实现另一条"检测重复"路径。这里 induction 头用途和最初被发现时(预测下一个 token)不同,而是当作 S1 的**位置信号**。
6. **Backup Name Mover Heads**:见 §4。

**第二段:量化验证(方法论贡献)**

- **Faithfulness**:$|F(M)-F(C)| = 0.46$,电路达完整模型 $87\%$ 性能。
- **Completeness**:光忠实不够(可能漏掉冗余通路)。定义:对任意子集 $K$,$|F(C\setminus K) - F(M\setminus K)|$ 都应小。随机/按类采样都显示"完整",但**贪心优化能找到不完整分数高达 $87\%$ 的 $K$**——作者诚实承认这是漏洞。
- **Minimality**:每个节点都得必要(存在某 $K$,去掉该节点明显掉分)。
- **对照**:一个"朴素电路"忠实度相近,但 completeness 一测就露馅。
- **对抗样本**:利用"模型靠重复检测区分 S/IO",构造让 IO 也重复的句子,把模型骗到 $23.4\%$ 时预测 S——证明理解有下游可用性。

## 4. 自修复在这篇里处于什么位置

**配角 / 意外副产品,不是主线。** 在 §4.4 "Backup Name Movers 的故事":

- 为检查"有没有漏掉 Name Mover 副本",一次性**敲掉全部 Name Mover 头**。出人意料——电路**几乎照常工作,logit 差只掉 $5\%$**。
- 事后再跑 path patching,发现原本"不干活"的 $8$ 个头(Backup Name Mover Heads)**顶了上来**,替代被敲掉头的功能。
- 作者假设:可能是**训练时用 dropout** 导致——模型被训练得对"部分组件失灵"有鲁棒性。并明确说"需要更多工作确定来源及是否普遍"。

这一段(加上 Negative Name Mover Heads)是后三篇的直接起点:**IOI 无意中记录下两个反直觉现象——敲掉主力头后有备份头补偿(self-repair 雏形)、以及有头专门朝反方向写(负向头)。McGrath / McDougall / Rushing 三篇本质上在追问"这两个现象到底是什么、为什么、有多普遍"。**

## 5. 与 TFM 迁移课题的关联

**可复用的技术/概念**:

- **Path patching + 干净/污染对照分布(p_IOI vs p_ABC)**:定位"某组件对某预测的因果作用"的通用范式。TFM 版需设计 tabular "污染分布"——如打乱某特征列、或替换支持集(in-context 样本)标签,抹掉目标信息但保留表结构。
- **均值消融而非零消融**:在 TFM 上大概率更重要,因表格模型激活尺度、隐含 bias 更敏感;零消融几乎肯定推到分布外。
- **Direct logit attribution / copy score 的类比**:"unembedding 方向"换成**分类头对某类别的方向**(TabPFN/TabICL 输出类别分布)。把组件输出投影到"预测类别 $c$"方向,定义 tabular 版 copy score。
- **faithfulness / completeness / minimality 三件套**:与任务无关的电路验证标准,几乎原样可搬——只要能定义标量度量 $F$(类比 logit difference,如"正确类 vs 竞争类的 logit 差")。
- **Backup 头 + dropout 假设**:直接指向课题——若 TFM 也有自修复,一个可检验假设是它是否源于训练时某种鲁棒化(dropout、特征遮蔽、bagging)。

**架构假设(务必在 TFM 上验证)**:

- ⚠️ **残差流线性可加**:全篇归因的地基。TabPFN/TabICL 是 transformer,原则上成立,但要确认无非标准跨层连接。**可验证**。
- ⚠️ **QK/OV 分解**:依赖标准多头注意力。TabICL 有"逐列/逐行"两级注意力,TabPFN 对特征和样本维分别注意——分解仍可做,但"head 从哪读、写什么"的语义要重新定义(读的是**样本**还是**特征**?)。
- ⚠️ **LayerNorm 位置**:作者注明 unembed 前有一个 LN,导致投影到 logit 方向"未被正确缩放",他们近似处理。TFM 若用 pre-LN / RMSNorm / 或分类头前有不同归一化,近似要重做。
- ⚠️ **"两个竞争 token 的 logit 差"度量**:强依赖 IOI 是"二选一名字"。TFM 预测是类别分布或回归值,要先想清楚 F 的 tabular 对应物(二分类天然对应,多分类/回归需另设计)。
- ⚠️ **"信息沿 token 位置流动"框架**:IOI 核心是跨序列位置搬运名字。TFM 里"位置"对应**支持集里哪个样本 / 哪个特征**,信息流几何完全不同——最需重新概念化的地方。

## 6. 阅读建议

- **精读**:Introduction(全貌)、§ Discovering the Circuit(各类头 + path patching 用法,方法内核)、§ Experimental validation 的 completeness 部分(方法论贡献 + 作者诚实自省)。
- **核心图**:图 1(电路全貌)、电路结构图、Name Mover 三联图(注意力 vs logit 投影)、completeness 散点图。
- **可跳读**:大量附录第一遍可跳;但 §4.4 Backup 头这一小节**对本课题要精读**,是整条线索源头。
- **前置知识**:decoder-only transformer 结构、注意力的 QK/OV 视角(Elhage 2021 直觉)、activation patching 基本思想。不需读 Elhage 全文,但要理解"残差流 + 头分解"这套语言。

> 待核对:摘要/正文对头的数量有轻微不一致("26 attention heads" vs 正文一处 "all 28 heads",及 "subsets of as few as 13")。引用确切数字时以正文表格/图为准。

---

## 后续阅读计划(暂缓,勿展开)

### 第二层:磨工具
- Edin et al. 2025, "GIM: Gradient Interaction Modifications" (arXiv 2505.17630)
- Lad et al. 2025, "Remarkable Robustness of LLMs: Stages of Inference"

### 第三层:TFM 战场
- Balef, Koshil & Eggensperger 2026, "Is One Layer Enough?" (arXiv 2605.06510)
- Morgan Stanley 2026, "A Mechanistic Study of Tabular Foundation Models" (arXiv 2605.21288)
