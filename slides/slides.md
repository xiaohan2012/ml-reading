---
theme: default
title: Do tabular foundation models repair themselves?
info: TFM self-repair — 2-minute talk
class: text-center
transition: slide-left
mdc: true
---

# Do tabular foundation models repair themselves?

<div class="byline">
  Han Xiao
  <div class="affil">BlueDot Impact &mdash; Technical AI Safety Project</div>
  <div class="affil">August 21, 2026</div>
</div>

<style scoped>
.byline { margin-top: 1.6rem; font-size: 1.1rem; }
.affil { font-size: 0.85rem; opacity: 0.7; margin-top: 0.3rem; }
</style>

<!--
My name is Han, a data scientist in Finland. I will talk about self-repair in tabular foundation models. Thanks for checking this video.
-->

---

# Background

<div class="lead-line">Language models <em>self-repair</em>: ablate one layer, later layers compensate for the damage</div>

<div class="lm-fig">

<div class="lm-col">
  <div class="cap">prompt</div>
  <div class="prompt">Honus Wagner professionally plays the sport of &hellip;</div>
</div>

<div class="arrow">&rarr;</div>

<div class="model-wrap">
  <div class="labels">
    <div class="lbl-sp"></div>
    <div class="lbl"></div>
    <div class="lbl fade" :class="{ on: $clicks >= 1 }">ablate layer 2 &rarr;</div>
    <div class="lbl fade" :class="{ on: $clicks >= 2 }">layer 3 repairs &rarr;</div>
    <div class="lbl"></div>
  </div>
  <div class="model">
    <div class="model-title">Transformer</div>
    <div class="blk">layer 1</div>
    <div class="blk" :class="{ abl: $clicks >= 1 }">layer 2</div>
    <div class="blk" :class="{ rep: $clicks >= 2 }">layer 3</div>
    <div class="blk">layer 4</div>
  </div>
</div>

<div class="arrow">&rarr;</div>

<div class="lm-col">
  <div class="cap">next token</div>
  <table class="cand">
    <thead>
      <tr><th></th><th>logit</th><th>prob</th></tr>
    </thead>
    <tbody>
      <tr class="hit">
        <td>baseball</td>
        <td><span class="old" :class="{ dim: $clicks >= 3 }">9.2</span><span class="post fade" :class="{ on: $clicks >= 3 }"> &rarr; 8.9</span></td>
        <td><span class="old" :class="{ dim: $clicks >= 3 }">0.71</span><span class="post fade" :class="{ on: $clicks >= 3 }"> &rarr; 0.66</span></td>
      </tr>
      <tr>
        <td>football</td>
        <td><span class="old" :class="{ dim: $clicks >= 3 }">7.4</span><span class="post fade" :class="{ on: $clicks >= 3 }"> &rarr; 7.5</span></td>
        <td><span class="old" :class="{ dim: $clicks >= 3 }">0.12</span><span class="post fade" :class="{ on: $clicks >= 3 }"> &rarr; 0.13</span></td>
      </tr>
      <tr class="dots">
        <td>⋯</td><td>⋯</td><td>⋯</td>
      </tr>
      <tr>
        <td>hockey</td>
        <td><span class="old" :class="{ dim: $clicks >= 3 }">6.9</span><span class="post fade" :class="{ on: $clicks >= 3 }"> &rarr; 7.0</span></td>
        <td><span class="old" :class="{ dim: $clicks >= 3 }">0.07</span><span class="post fade" :class="{ on: $clicks >= 3 }"> &rarr; 0.08</span></td>
      </tr>
    </tbody>
  </table>
</div>

</div>

<div class="lm-note">
  <div class="fade strong" :class="{ on: $clicks >= 3 }">the output barely moves</div>
</div>

<div class="rq fade" :class="{ on: $clicks >= 4 }"><b>Research question</b>: do tabular foundation models repair themselves as well?</div>

<div class="click-anchors">
  <span v-click></span><span v-click></span>
  <span v-click></span><span v-click></span>
</div>

<style scoped>
.click-anchors { font-size: 0; line-height: 0; height: 0; }
.fade { opacity: 0; transition: opacity 0.35s ease; }
.fade.on { opacity: 1; }
.lead-line { font-size: 1.05rem; }
.rq { text-align: center; font-size: 1.4rem; margin-top: 1rem; }
.lm-fig {
  display: flex; align-items: center; justify-content: center;
  gap: 1.6rem; margin-top: 1.4rem;
}
.lm-col { display: flex; flex-direction: column; align-items: center; }
.cap { font-size: 0.8rem; opacity: 0.6; margin-bottom: 0.35rem; }
.prompt {
  max-width: 13rem; font-size: 0.9rem; line-height: 1.5;
  border: 1px solid currentColor; border-radius: 0.35rem;
  padding: 0.5rem 0.7rem;
}
.arrow { font-size: 1.6rem; opacity: 0.5; }
.model-wrap { display: flex; align-items: flex-start; }
.labels {
  display: flex; flex-direction: column; align-items: flex-end;
  padding-top: 0.7rem; margin-right: 0.5rem;
}
.lbl-sp { height: 1.76rem; }
.lbl {
  height: 1.7rem; margin: 0.18rem 0;
  display: flex; align-items: center;
  font-size: 0.8rem; white-space: nowrap;
}
.model {
  display: flex; flex-direction: column; align-items: stretch;
  border: 1px solid currentColor; border-radius: 0.5rem; padding: 0.7rem 1rem;
}
.model-title {
  font-size: 0.9rem; line-height: 1.4; margin-bottom: 0.5rem; text-align: center;
}
.blk {
  position: relative; box-sizing: border-box;
  height: 1.7rem; margin: 0.18rem 0; padding: 0 1.3rem;
  display: flex; align-items: center; justify-content: center;
  border: 1px solid currentColor; border-radius: 0.25rem;
  font-size: 0.78rem; opacity: 0.75;
  transition: all 0.35s ease;
}
.blk.abl {
  opacity: 1; color: rgba(120,120,120,0.55); border-color: #ef4444;
}
.blk.abl::before, .blk.abl::after {
  content: ''; position: absolute; left: 4%; right: 4%; top: 50%;
  height: 2px; background: #ef4444;
}
.blk.abl::before { transform: rotate(11deg); }
.blk.abl::after { transform: rotate(-11deg); }
.blk.rep {
  opacity: 1; color: #16a34a; border-color: #16a34a;
  background: rgba(34,197,94,0.15);
}
.cand { border-collapse: collapse; font-size: 0.82rem; }
.cand th, .cand td {
  padding: 0.22rem 0.6rem; text-align: right;
  border-bottom: 1px solid rgba(128,128,128,0.25);
}
.cand td:first-child, .cand th:first-child { text-align: left; }
.cand th { font-weight: 600; opacity: 0.6; font-size: 0.72rem; }
.cand tr.hit td { font-weight: 700; }
.cand tr.dots td { opacity: 0.5; }
.cand .old { transition: color 0.35s ease; }
.cand tr td .old.dim { color: #9ca3af; }
.lm-note {
  text-align: center; font-size: 0.95rem; line-height: 1.9; margin-top: 1.4rem;
}
.lm-note .strong { font-weight: 700; }
</style>

<!--
Self-repair is widely observed on large language models.

[click] In a nutshell, abalting one layer may cause "damage" to the computation



[click] but later layer repairs the damage by changing its behaviour.

[click] As a consequence, the output logits barely move. This is called self-repair (or coined Hydra effect in the literature)

[click] In this project, we ask whether self-repair also exists in different type of models called tabular foundation models
-->

---

# What is a tabular foundation model (TFM)?

- TFMs $\approx$ transformers for learning on tabular data

<div class="tfm-fig">

<div class="col">
  <div class="fade" :class="{ on: $clicks >= 1 }">
    <div class="cap">context table</div>
    <table class="tbl">
      <tr><td>x<sub>11</sub></td><td>⋯</td><td>x<sub>1m</sub></td><td class="y">y<sub>1</sub></td></tr>
      <tr><td>x<sub>21</sub></td><td>⋯</td><td>x<sub>2m</sub></td><td class="y">y<sub>2</sub></td></tr>
      <tr><td>⋯</td><td>⋯</td><td>⋯</td><td class="y">⋯</td></tr>
      <tr><td>x<sub>n1</sub></td><td>⋯</td><td>x<sub>nm</sub></td><td class="y">y<sub>n</sub></td></tr>
    </table>
  </div>
  <div class="fade" :class="{ on: $clicks >= 2 }">
    <div class="cap cap-lo">test table</div>
    <table class="tbl">
      <tr><td>x<sub>11</sub></td><td>⋯</td><td>x<sub>1m</sub></td><td class="y q">?</td></tr>
      <tr><td>⋯</td><td>⋯</td><td>⋯</td><td class="y q">⋯</td></tr>
      <tr><td>x<sub>k1</sub></td><td>⋯</td><td>x<sub>km</sub></td><td class="y q">?</td></tr>
    </table>
  </div>
</div>

<div class="arrow fade" :class="{ on: $clicks >= 3 }">&rarr;</div>

<div class="model fade" :class="{ on: $clicks >= 3 }">
  <div class="model-title">Transformer</div>
  <div class="blk">block</div>
  <div class="blk">block</div>
  <div class="dots">⋮</div>
  <div class="blk">block</div>
</div>

<div class="arrow fade" :class="{ on: $clicks >= 4 }">&rarr;</div>

<div class="col fade" :class="{ on: $clicks >= 4 }">
  <div class="cap">prediction</div>
  <table class="tbl">
    <tr><td class="y hat">ŷ<sub>1</sub></td></tr>
    <tr><td class="y hat">⋯</td></tr>
    <tr><td class="y hat">ŷ<sub>k</sub></td></tr>
  </table>
</div>

</div>

<div class="fade" :class="{ on: $clicks >= 5 }">

- TFMs do *in-context learning*, i.e., there is **no** gradient update

</div>

<div class="click-anchors">
  <span v-click></span><span v-click></span><span v-click></span>
  <span v-click></span><span v-click></span>
</div>

<style scoped>
.tfm-fig {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1.6rem;
  margin: 2rem 0 1.5rem;
}
.col { display: flex; flex-direction: column; align-items: center; }
.cap { font-size: 0.8rem; opacity: 0.6; margin-bottom: 0.35rem; }
.cap-lo { margin-top: 1.2rem; }
.tbl { border-collapse: collapse; font-size: 0.85rem; }
.tbl td {
  border: 1px solid currentColor;
  padding: 0.25rem 0.6rem;
  text-align: center;
  min-width: 2.6rem;
}
.tbl td.y { border-left-width: 2px; background: rgba(59,130,246,0.12); }
.tbl td.q { color: #ef4444; font-weight: 600; }
.tbl td.hat { background: rgba(34,197,94,0.18); font-weight: 600; }
.arrow { font-size: 1.6rem; opacity: 0.5; }
.model {
  display: flex; flex-direction: column; align-items: center;
  border: 1px solid currentColor; border-radius: 0.5rem;
  padding: 0.7rem 1rem;
}
.model-title { font-size: 0.9rem; margin-bottom: 0.5rem; }
.blk {
  border: 1px solid currentColor; border-radius: 0.25rem;
  padding: 0.2rem 1.2rem; margin: 0.15rem 0;
  font-size: 0.75rem; opacity: 0.75;
}
.dots { font-size: 0.9rem; opacity: 0.5; line-height: 1; }
.fade { opacity: 0; transition: opacity 0.35s ease; }
.fade.on { opacity: 1; }
.click-anchors { font-size: 0; line-height: 0; height: 0; }
</style>

<!--
Tabular foundation model is essentially transformers for making predictions on tabular data

[click] The model operates on two data tables of the same schema.

Context table: which contains both the data attributes and target labels

[click] and test table: which contains only data attributes but misses the labels

[click] the model is notably transformer-based, sharing the similar technical backbone as language models

[click] after ingesting the two tables, the model predicts the missing labels on the test data based on the context

[click] TMFs do in-context learning (a prevalent technique used in language models), in other words, the model is never fine-tuned and does not do any gradient update
-->

---

# TFM versus supervised ML and language models

<div class="cmp" :class="$clicks >= 2 ? 's3' : ($clicks >= 1 ? 's2' : 's1')">
  <table class="cmp-tbl">
    <thead>
      <tr>
        <th></th><th class="c-sml">Supervised ML</th>
        <th class="c-tfm">TFM</th><th class="c-lm">Language model</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>Input</td><td class="c-sml">table</td><td class="c-tfm">table</td><td class="c-lm">text sequence</td></tr>
      <tr><td>Per-task training</td><td class="c-sml">yes</td><td class="c-tfm">no</td><td class="c-lm">no</td></tr>
      <tr><td>How it predicts</td><td class="c-sml">fitted model</td><td class="c-tfm">in-context</td><td class="c-lm">in-context</td></tr>
      <tr><td>Architecture</td><td class="c-sml">trees / MLP</td><td class="c-tfm">transformer</td><td class="c-lm">transformer</td></tr>
      <tr><td>Example</td><td class="c-sml">XGBoost</td><td class="c-tfm">TabPFN-v2, Mitra</td><td class="c-lm">GPT</td></tr>
    </tbody>
  </table>
</div>

<div class="cmp-sum" :class="{ on: $clicks >= 2 }">TFM = the problem domain of supervised ML, the mechanism of a language model.</div>

<div class="click-anchors">
  <span v-click></span><span v-click></span>
  <span v-click></span><span v-click></span>
</div>

<style scoped>
.click-anchors { font-size: 0; line-height: 0; height: 0; }
.cmp { width: fit-content; margin: 1.2rem auto 0; }
.cmp-tbl { border-collapse: collapse; font-size: 1.05rem; }
.cmp-tbl th, .cmp-tbl td {
  padding: 0.5rem 1.6rem; text-align: left;
  border-bottom: 1px solid rgba(128,128,128,0.3);
}
.cmp-tbl th { font-weight: 700; }
.cmp-tbl td:first-child { font-weight: 700; }
.cmp-tbl .c-sml, .cmp-tbl .c-lm, .cmp-tbl .c-tfm {
  transition: opacity 0.3s ease;
}
.cmp.s1 .c-lm { opacity: 0; }
.cmp.s2 .c-sml { opacity: 0; }

.cmp-sum {
  text-align: center; font-size: 1.25rem;
  margin-top: 1.4rem;
  opacity: 0; transition: opacity 0.3s ease;
}
.cmp-sum.on { opacity: 1; }
</style>

<!--
TFMs are closely related to supervied machine learning and language models

[click] compared to supervised ML for tabular data, tabular models consumes the same data format and highly overlap on practical applications (predicting on tabular data)
the key difference lies on whether any model parameter update is involved. For supervised ML, a new model needs to be trained for each task, whereas tabular models do not require that

[click] compared the language models, the main difference lies in the data format.

In contrast to tabular models, Language models predicts next word on text sequence. However, their underlying technology is closely related -- they both use transformer-based architecture and do in-context learning.
-->


---

# Why the RQ matters for AI safety

- TFMs are deployed in high-stake domains, e.g., lending: $y = 1$ grant the loan, $y = 0$ deny
- Ablation is a main tool for understanding how a model works &mdash; and self-repair breaks it

<div class="sf fade" :class="{ on: $clicks >= 1 }">

<div class="col">
  <div class="cap">context table</div>
  <table class="tbl">
    <tr><td>x<sub>11</sub></td><td>⋯</td><td>x<sub>1m</sub></td><td class="y">1</td></tr>
    <tr><td>x<sub>21</sub></td><td>⋯</td><td>x<sub>2m</sub></td><td class="y">0</td></tr>
    <tr><td>⋯</td><td>⋯</td><td>⋯</td><td class="y">⋯</td></tr>
    <tr><td>x<sub>n1</sub></td><td>⋯</td><td>x<sub>nm</sub></td><td class="y">1</td></tr>
  </table>
  <div class="cap cap-lo">test table</div>
  <table class="tbl">
    <tr><td>x<sub>11</sub></td><td>⋯</td><td>x<sub>1m</sub></td><td class="y q">?</td></tr>
    <tr><td>x<sub>21</sub></td><td>⋯</td><td>x<sub>2m</sub></td><td class="y q">?</td></tr>
  </table>
</div>

<div class="arrow">&rarr;</div>

<div class="model">
  <div class="model-title">Transformer</div>
  <div class="blk">layer 1</div>
  <div class="blk" :class="$clicks >= 3 ? 'abl' : ($clicks >= 2 ? 'imp blinking' : 'imp')">layer 2</div>
  <div class="blk" :class="{ comp: $clicks >= 3, blinking: $clicks >= 6 }">layer 3</div>
  <div class="blk">layer 4</div>
</div>

<div class="arrow">&rarr;</div>

<div class="col">
  <div class="cap">prediction</div>
  <div class="pred-box" :class="{ flash: $clicks >= 4 }">
    <table class="tbl">
      <tr><td class="hat">ŷ<sub>1</sub> = 1</td></tr>
      <tr><td class="hat">ŷ<sub>2</sub> = 0</td></tr>
    </table>
  </div>
  <div class="same" :class="{ flash: $clicks >= 4 }">unchanged</div>
</div>

</div>

<div class="verdict-wrap">
  <div class="verdict" :class="{ shown: $clicks >= 5, struck: $clicks >= 7 }">
    &ldquo;Layer 2 does not contribute.&rdquo;
  </div>
</div>

<div class="click-anchors">
  <span v-click></span><span v-click></span><span v-click></span><span v-click></span>
  <span v-click></span><span v-click></span><span v-click></span>
</div>

<style scoped>
.click-anchors { font-size: 0; line-height: 0; height: 0; }
.fade { opacity: 0; transition: opacity 0.35s ease; }
.fade.on { opacity: 1; }
.sf {
  display: flex; align-items: center; justify-content: center;
  gap: 1.6rem; margin: 1.6rem 0 0.8rem;
}
.col { display: flex; flex-direction: column; align-items: center; }
.cap { font-size: 0.8rem; opacity: 0.6; margin-bottom: 0.35rem; }
.cap-lo { margin-top: 1.2rem; }
.tbl { border-collapse: collapse; font-size: 0.85rem; }
.tbl td {
  border: 1px solid currentColor; padding: 0.25rem 0.6rem;
  text-align: center; min-width: 2.6rem;
}
.tbl td.y { border-left-width: 2px; background: rgba(59,130,246,0.12); }
.tbl td.q { color: #ef4444; font-weight: 600; }
.tbl td.hat { background: rgba(34,197,94,0.18); font-weight: 600; }
.arrow { font-size: 1.6rem; opacity: 0.5; }
.model {
  display: flex; flex-direction: column; align-items: stretch;
  border: 1px solid currentColor; border-radius: 0.5rem; padding: 0.7rem 1rem;
}
.model-title { font-size: 0.9rem; margin-bottom: 0.5rem; text-align: center; }
.blk {
  border: 1px solid currentColor; border-radius: 0.25rem;
  padding: 0.2rem 1.2rem; margin: 0.15rem 0;
  font-size: 0.75rem; opacity: 0.75; text-align: center;
  transition: all 0.4s ease;
}
.blk.imp {
  opacity: 1; color: #16a34a; border-color: #16a34a;
  background: rgba(34,197,94,0.15);
}
.blk.imp.blinking { animation: blink-comp 0.5s ease-in-out 2; }
.blk.abl {
  opacity: 1; color: #ef4444; border-color: #ef4444;
  background: rgba(239,68,68,0.15); text-decoration: line-through;
}
.blk.comp {
  opacity: 1; color: #16a34a; border-color: #16a34a;
  background: rgba(34,197,94,0.15);
}
.blk.comp.blinking { animation: blink-comp 0.5s ease-in-out 4; }
@keyframes blink-comp {
  0%, 100% { background: rgba(34,197,94,0.15); }
  50%      { background: rgba(34,197,94,0.7); }
}
.pred-box {
  border: 2px dashed transparent; border-radius: 0.35rem; padding: 0.3rem;
}
.pred-box.flash {
  border-color: #16a34a;
  animation: blink-border 0.5s ease-in-out 4;
}
@keyframes blink-border {
  0%, 100% { border-color: #16a34a; }
  50%      { border-color: transparent; }
}
.same {
  margin-top: 0.4rem; font-size: 0.8rem; font-weight: 600;
  color: #16a34a; opacity: 0;
}
.same.flash { opacity: 1; animation: blink 0.5s ease-in-out 4; }
@keyframes blink {
  0%, 100% { opacity: 1; }
  50%      { opacity: 0.15; }
}
.verdict-wrap { text-align: center; margin-top: 0.4rem; }
.verdict {
  display: inline-block; position: relative;
  font-weight: 600; font-size: 1rem;
  opacity: 0; transition: opacity 0.3s ease;
}
.verdict.shown { opacity: 1; }
.verdict::after {
  content: ''; position: absolute; left: 0; top: 50%;
  height: 3px; background: #ef4444; border-radius: 2px;
  width: 0; transition: width 0.45s ease-out;
}
.verdict.struck::after { width: 100%; }
</style>

<!--
TFMs are being deployed in high-stake scenarios, for example, loan lending, in which the models decide whether a loan is granted to an applicant or not.

Model transparency and trust-worthiness matters. In order to improve transparency, one needs to open up model and understand how it works.

Ablation is a widely adopted technique to gain such understanding.


[click] As we show next, self-repair can break the conclusion.

[click] In this toy example, layer 2 plays a key role in shaping the final prediction, when nothing is ablated on

[click] We can ablate on it, however, layer 3 repairs the damage

[click] and the output logits barely change

[click] so one concludes that layer 2 do not make a difference

[click] but the reality is that layer 2 does the compensation

[click] in other words, self-repair leads to a seemingly plausible conclusion, which is wrong
-->

---

# Self-repair: definition

<div class="eff-wrap">
  <img src="/hydra-te-de-ie.png" class="w-full" />
  <svg class="eff-overlay" viewBox="0 0 4206 1438">
    <defs>
      <marker id="ar" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
        <path d="M0,0 L0,6 L6,3 z" fill="#2563eb" />
      </marker>
    </defs>
    <g class="on layer-lab">
      <text x="350" y="515">layer B</text>
      <text x="350" y="965">layer A</text>
    </g>
    <rect class="box" :class="{ on: $clicks === 1 }" x="700" y="190" width="400" height="1240" />
    <rect class="box" :class="{ on: $clicks === 2 || $clicks === 3 }" x="1270" y="190" width="930" height="1240" />
    <rect class="box" :class="{ on: $clicks === 4 || $clicks === 5 }" x="2220" y="190" width="800" height="1240" />
    <rect class="box" :class="{ on: $clicks === 6 }" x="3120" y="190" width="1060" height="1240" />
    <g :class="{ on: $clicks === 1 }">
      <text x="690" y="130">clean run</text>
    </g>
    <g :class="{ on: $clicks === 3 }">
      <text x="1150" y="320">B reacts freely</text>
      <line x1="1480" y1="355" x2="1680" y2="455" marker-end="url(#ar)" />
    </g>
    <g :class="{ on: $clicks === 5 }">
      <text x="2000" y="430">B held clean</text>
      <line x1="2330" y1="465" x2="2400" y2="590" marker-end="url(#ar)" />
    </g>
    <g :class="{ on: $clicks === 6 }">
      <text x="3100" y="140">B's reaction only</text>
      <line x1="3450" y1="180" x2="3580" y2="420" marker-end="url(#ar)" />
    </g>
  </svg>
</div>

<div class="defs">
  <div class="lead"><b>Effect</b> = &Delta;<i> y</i> caused by an intervention</div>
  <ul>
    <li :class="{ on: $clicks >= 2 }"><b>Total effect (TE)</b>: ablate A, let B react freely</li>
    <li :class="{ on: $clicks >= 4 }"><b>Direct effect (DE)</b>: ablate A, freeze B at its clean value</li>
    <li :class="{ on: $clicks >= 6 }"><b>Indirect effect (IE)</b>: B's direct effect due to the ablation of A</li>
    <li :class="{ on: $clicks >= 7 }"><b>Compensation effect (CE)</b> = DE &minus; TE = &minus;IE: how much B compensates for the ablation of A</li>
  </ul>
  <div class="verdict" :class="{ on: $clicks >= 8 }">
    <b>self-repair</b> &nbsp;&hArr;&nbsp; <b>CE &gt; 0</b> <span class="qual">(consistently)</span>
  </div>
</div>

<div class="click-anchors">
  <span v-click></span><span v-click></span><span v-click></span>
  <span v-click></span><span v-click></span><span v-click></span>
  <span v-click></span><span v-click></span>
</div>

<style scoped>
.eff-wrap { position: relative; width: 72%; margin: 0.4rem auto 0; }
.eff-overlay {
  position: absolute; inset: 0; width: 100%; height: 100%;
  pointer-events: none;
}
.eff-overlay g { opacity: 0; transition: opacity 0.3s ease; }
.eff-overlay g.on { opacity: 1; }
.eff-overlay text {
  fill: #2563eb; font-size: 93px; font-weight: 700;
}
.eff-overlay line { stroke: #2563eb; stroke-width: 9; }
.eff-overlay .layer-lab text { fill: currentColor; }
.eff-overlay .box {
  fill: none; stroke: #2563eb; stroke-width: 8; stroke-dasharray: 26 20;
  rx: 24; opacity: 0; transition: opacity 0.3s ease;
}
.eff-overlay .box.on { opacity: 1; animation: box-blink 0.5s ease-in-out 2; }
@keyframes box-blink {
  0%, 100% { opacity: 1; }
  50%      { opacity: 0.1; }
}
.eff-legend {
  text-align: center; font-size: 0.7rem; opacity: 0.6; margin-bottom: 0.8rem;
}
.defs { font-size: 0.95rem; line-height: 1.9; margin-top: 1.6rem; }
.defs .lead { margin-bottom: 0.4rem; }
.defs ul { list-style: disc; padding-left: 1.4rem; }
.defs li { opacity: 0; transition: opacity 0.3s ease; }
.defs li.on { opacity: 1; }
.defs .verdict {
  margin-top: 0.7rem; text-align: center; font-size: 1.15rem;
  color: #16a34a; opacity: 0; transition: opacity 0.3s ease;
}
.defs .verdict.on { opacity: 1; }
.defs .verdict .qual { font-weight: 400; }
.click-anchors { font-size: 0; line-height: 0; height: 0; }
</style>

<!--
To answer our research question, let's define what self-repair is and how to quantify it.

The definition boils down to a few kinds of effects. We measure the effect of an intervention by the change of output logit

[click] We first run the model without any intervention (called clean run), we save the internal activations for later use

[click] Suppose we want to intervene on layer A by substituting the activations with a corrupted value.


[click] In the case of total effect, we let the downstream layer B react freely and measure the final effect as the total effect

[click] In direct effect, B uses the clean values we obtained from the clean run.

[click] Intuitively, only the A-to-y path carries the intervention.

[click] Indirect effect is the completement, it measures B's own contribution through the B -> y, as if A is abalted on

[click] finally we define compensation effect as direct effect minus total effect, which measures how much B compensates for the ablation of A

[click] definition of self-repair is therefore, compensation effect is larger than 0 consistently, under different inputs
-->

---

# Why both TE and DE are needed

- A two-layer transformer, layer $A$ and layer $B$
- Layer $A$ constantly writes $a = 1$

<div class="fade" :class="{ on: $clicks >= 1 }">

* The intervention is $do(A = 0)$
* The output $y$ is defined as

$$
y = \begin{cases}
1 & \text{if } a + b > 0 \\
0 & \text{otherwise}
\end{cases}
$$

</div>

<div class="toy fade" :class="{ on: $clicks >= 2, s1: $clicks === 3, s2: $clicks === 4, s3: $clicks >= 5 }">
  <table class="toy-tbl">
    <thead>
      <tr>
        <th>Scenario</th><th><i>B</i></th><th><i>a</i> + <i>b</i></th><th><i>y</i></th>
        <th class="c-te">TE</th><th class="c-de">DE</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Redundant</td><td><i>b</i> = 1</td><td>2</td><td>1</td>
        <td class="c-te">0</td><td class="c-de">0</td>
      </tr>
      <tr>
        <td>Repaired</td><td><i>b</i> = 1 &minus; <i>a</i></td><td>1</td><td>1</td>
        <td class="c-te">0</td><td class="c-de">1</td>
      </tr>
      <tr>
        <td>Load-bearing</td><td><i>b</i> = 0</td><td>1</td><td>1</td>
        <td class="c-te">1</td><td class="c-de">1</td>
      </tr>
    </tbody>
  </table>
  <div class="ann ann1">TE alone: redundant = repaired &nbsp;&#10060;</div>
  <div class="ann ann2">DE alone: repaired = load-bearing &nbsp;&#10060;</div>
  <div class="ann ann3">TE and DE: all three separated &nbsp;&#9989;</div>
</div>

<div class="click-anchors">
  <span v-click></span><span v-click></span><span v-click></span>
  <span v-click></span><span v-click></span>
</div>

<style scoped>
.click-anchors { font-size: 0; line-height: 0; height: 0; }
.fade { opacity: 0; transition: opacity 0.35s ease; }
.fade.on { opacity: 1; }
.katex-display { font-size: 0.85em; }
.toy {
  position: relative; width: fit-content;
  margin: 0.6rem auto 0;
}
.toy-tbl { border-collapse: collapse; font-size: 1.2rem; }
.toy-tbl th, .toy-tbl td {
  padding: 0.45rem 1.5rem; text-align: center;
  border-bottom: 1px solid rgba(128,128,128,0.3);
}
.toy-tbl th:first-child, .toy-tbl td:first-child { text-align: left; }
.toy-tbl th { font-weight: 700; }
.toy-tbl td:first-child { font-weight: 700; }
.toy-tbl th.c-te, .toy-tbl td.c-te,
.toy-tbl th.c-de, .toy-tbl td.c-de {
  border-left: 2px dashed transparent; border-right: 2px dashed transparent;
  transition: border-color 0.3s ease;
}
.toy-tbl th.c-te, .toy-tbl th.c-de { border-top: 2px dashed transparent; }
.toy.s1 th.c-te, .toy.s1 td.c-te { border-left-color: #9ca3af; border-right-color: #9ca3af; }
.toy.s1 th.c-te { border-top-color: #9ca3af; }
.toy.s1 .toy-tbl tbody tr:last-child td.c-te { border-bottom: 2px dashed #9ca3af !important; }
.toy.s2 th.c-de, .toy.s2 td.c-de { border-left-color: #9ca3af; border-right-color: #9ca3af; }
.toy.s2 th.c-te, .toy.s2 td.c-te { border-right-color: #9ca3af; }
.toy.s2 th.c-de { border-top-color: #9ca3af; }
.toy.s2 .toy-tbl tbody tr:last-child td.c-de { border-bottom: 2px dashed #9ca3af !important; }
.toy.s3 th.c-te, .toy.s3 td.c-te { border-left-color: #9ca3af; }
.toy.s3 th.c-de, .toy.s3 td.c-de { border-right-color: #9ca3af; }
.toy.s3 th.c-te, .toy.s3 th.c-de { border-top-color: #9ca3af; }
.toy.s3 .toy-tbl tbody tr:last-child td.c-te,
.toy.s3 .toy-tbl tbody tr:last-child td.c-de { border-bottom: 2px dashed #9ca3af !important; }
.ann {
  position: absolute; top: 100%; left: 50%; transform: translateX(-50%);
  margin-top: 0.5rem; font-size: 1.05rem; font-weight: 600;
  white-space: nowrap; opacity: 0; transition: opacity 0.3s ease;
}
.ann1, .ann2 { color: #ef4444; }
.ann3 { color: #16a34a; }
.toy.s1 .ann1 { opacity: 1; }
.toy.s2 .ann2 { opacity: 1; }
.toy.s3 .ann3 { opacity: 1; }
</style>


<!--
So why both TE and DE are needed to study self-repair? Here is a toy example for illustration

Consider a two layer transformer, with layer A and layer B. A is constant function, always outputing value 1, meanwhile it is the layer we ablate on

[click] The model predicts 1 if summation of the two layer's output is greater than 1, otherwise, 0 is predicted

[click] There are two possiuble scenarios, layer B is redundant or self-repairs

[click] if we measure total effect only, the two scenarios are indistinguishable

[click] to tell them apart, we have to consider both TE and DE
-->

---

# Main results: language vs. tabular models

<div class="res">

<div class="res-col">
  <div class="res-cap">Language model (Chinchilla 7B)</div>
  <div class="fig-wrap">
    <img src="/hydra-de-te.png" />
    <svg class="fig-overlay" viewBox="0 0 734 708">
      <rect class="patch" x="190" y="650" width="380" height="58" />
      <text class="axlab" x="380" y="692">DE (direct effect)</text>
      <rect class="patch" x="0" y="150" width="52" height="440" />
      <text class="axlab" x="26" y="370" transform="rotate(-90 26 370)">TE (total effect)</text>
      <ellipse class="hl" :class="{ on: $clicks >= 1 }" cx="430" cy="440" rx="150" ry="95" transform="rotate(-38 430 440)" />
      <text class="note-in note-lm" :class="{ on: $clicks >= 1 }" x="706" y="596" text-anchor="end">DE &gt; TE &rarr; CE &gt; 0</text>
    </svg>
  </div>
  <div class="res-note good" :class="{ on: $clicks >= 1 }">strong sign of self-repair</div>
</div>

<div class="res-col">
  <div class="res-cap fade" :class="{ on: $clicks >= 2 }">Tabular foundation model (Mitra)</div>
  <div class="fig-wrap fade" :class="{ on: $clicks >= 2 }">
    <img src="/mitra-de-te.png" />
    <svg class="fig-overlay" viewBox="0 0 1102 1064">
      <ellipse class="hl" :class="{ on: $clicks >= 3 }" cx="740" cy="514" rx="190" ry="105" transform="rotate(-38 740 514)" />
      <text class="note-in note-tfm" :class="{ on: $clicks >= 3 }" x="760" y="770" text-anchor="middle">almost empty</text>
    </svg>
  </div>
  <div class="res-note bad" :class="{ on: $clicks >= 3 }">no evidence of self-repair</div>
  <div class="res-tag fade" :class="{ on: $clicks >= 3 }">&uarr; our finding!</div>
</div>

</div>

<div class="foot fade" :class="{ on: $clicks >= 3 }">
  The two axes are not on a common scale: the language model reports a logit difference, the TFM a margin. Only the position relative to the diagonal is comparable.
</div>

<div class="click-anchors">
  <span v-click></span><span v-click></span><span v-click></span>
</div>

<style scoped>
.res { display: flex; gap: 2.5rem; justify-content: center; margin-top: 0.6rem; }
.res-col { display: flex; flex-direction: column; align-items: center; width: 42%; }
.res-cap { font-size: 0.85rem; font-weight: 600; margin-bottom: 0.3rem; }
.fig-wrap { position: relative; width: 100%; }
.fig-wrap img { width: 100%; }
.fig-overlay { position: absolute; inset: 0; width: 100%; height: 100%; }
.fig-overlay .patch { fill: white; }
.fig-overlay .axlab {
  fill: #111; font-size: 26px; font-weight: 600; text-anchor: middle;
}
.fig-overlay .hl {
  fill: none; stroke: #9ca3af; stroke-width: 5; stroke-dasharray: 14 10;
  opacity: 0; transition: opacity 0.3s ease;
}
.fig-overlay .hl.on { opacity: 1; }
.fig-overlay .note-in {
  fill: #111; font-weight: 700;
  opacity: 0; transition: opacity 0.3s ease;
}
.fig-overlay .note-in.on { opacity: 1; }
.fig-overlay .note-lm { font-size: 26px; }
.fig-overlay .note-tfm { font-size: 39px; }
.res-tag {
  color: #111; font-weight: 700; font-size: 1.1rem; margin-top: 0.2rem;
}
.fade, .res-note { opacity: 0; transition: opacity 0.3s ease; }
.fade.on, .res-note.on { opacity: 1; }
.click-anchors { font-size: 0; line-height: 0; height: 0; }
.res-note {
  font-size: 1.25rem; font-weight: 700; margin-top: 0.5rem;
  white-space: nowrap;
}
.res-note.good { color: #16a34a; }
.res-note.bad { color: #ef4444; }
.foot { font-size: 0.65rem; color: #9ca3af; margin-top: 0.8rem; }
</style>



<!--
Here is our finding. Evidence of self-repair is weak in TFMs.

For reference, we first illustrate the case of language models, where evidence self-repair is strong

[click] Each dot represents a (layer, input) pair, where input is some text sequence. X and Y axises represent DE and TE respectively

[click] Points below the diagonal line correspond to the cases where self-repair happens, because DE > TE, therefore CE is positive.

For the language model under study, self-repair consistently appears

[click] for tabular models, the lower diagonl is almost empty, indicating self-repair rarely exists

Here, we show only one model because other models have similar results.
-->

---

# Want to know more?

<div class="link-wrap">
  <a href="https://xiaohan2012.github.io/articles/tfm-self-repair/">
    xiaohan2012.github.io/articles/tfm-self-repair
  </a>
</div>

<style scoped>
.link-wrap {
  display: flex; align-items: center; justify-content: center;
  height: 60vh;
}
.link-wrap a {
  font-size: 1.9rem; font-weight: 600;
  color: inherit; text-decoration: none;
}
</style>

<!--
If you want to know more, please check out the full article
-->
