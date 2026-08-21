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
.byline { margin-top: 1.8rem; font-size: 1.5rem; }
.affil { font-size: 1.15rem; opacity: 0.7; margin-top: 0.45rem; }
</style>

<!--
My name is Han, a data scientist based in Finland. I will talk about self-repair in tabular foundation models.
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

<div class="rq-cite fade" :class="{ on: $clicks >= 4 }">
  Balef et al., <em>Is One Layer Enough? Understanding Inference Dynamics in Tabular Foundation Models</em>,
  ICML 2026 &mdash; report that they do.
</div>

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
.rq-cite {
  text-align: center; font-size: 0.72rem; margin-top: 0.4rem;
}
.rq-cite.on { opacity: 0.6; }
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
Self-repair is widely observed in large language models.

Consider the task of next token prediction. In this example, without any intervention, the model predicts 'baseball' with a logit value of 9.2.

[click] Ablating one layer may cause "damage" to the computation,

[click] In a model with self-repair, a later layer compensates for the damage by changing its behaviour.

[click] As a consequence, the output logits barely move. This is called self-repair.

[click] In this project, we ask whether self-repair also exists in a different type of model, called tabular foundation models. A recent paper reports that it does, but our results suggest the opposite.
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
  <div class="fade" :class="{ on: $clicks >= 1 }">
    <div class="cap cap-lo">test table</div>
    <table class="tbl">
      <tr><td>x<sub>11</sub></td><td>⋯</td><td>x<sub>1m</sub></td><td class="y q">?</td></tr>
      <tr><td>⋯</td><td>⋯</td><td>⋯</td><td class="y q">⋯</td></tr>
      <tr><td>x<sub>k1</sub></td><td>⋯</td><td>x<sub>km</sub></td><td class="y q">?</td></tr>
    </table>
  </div>
</div>

<div class="arrow fade" :class="{ on: $clicks >= 2 }">&rarr;</div>

<div class="model fade" :class="{ on: $clicks >= 2 }">
  <div class="model-title">Transformer</div>
  <div class="blk">block</div>
  <div class="blk">block</div>
  <div class="dots">⋮</div>
  <div class="blk">block</div>
</div>

<div class="arrow fade" :class="{ on: $clicks >= 2 }">&rarr;</div>

<div class="col fade" :class="{ on: $clicks >= 2 }">
  <div class="cap">prediction</div>
  <table class="tbl">
    <tr><td class="y hat">ŷ<sub>1</sub></td></tr>
    <tr><td class="y hat">⋯</td></tr>
    <tr><td class="y hat">ŷ<sub>k</sub></td></tr>
  </table>
</div>

</div>

<div class="fade" :class="{ on: $clicks >= 3 }">

- TFMs do *in-context learning*, i.e., there is **no** gradient update
- Same tasks as supervised ML, same mechanism as a language model

</div>

<div class="click-anchors">
  <span v-click></span><span v-click></span><span v-click></span>
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
A tabular foundation model is a transformer that predicts on tabular data.

[click] It reads two tables with the same schema: a context table with attributes and labels, and a test table where the labels are missing.

[click] One frozen transformer reads both and predicts the missing labels.

[click] This is in-context learning, the same mechanism language models use: no fine-tuning, no gradient update. So a TFM solves the problems of supervised machine learning with the mechanism of a language model.
-->

---
hide: true
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
So why studying self-repair in tabular foundation model matters for AI safety.

TFMs are being increasingly deployed in high-stake settings, for example loan lending, where the model decides whether an applicant is granted a loan.

Transparency and trustworthiness matter there. To improve transparency, we need to open up the model and understand how it works.

In particular, ablation is a widely used technique for gaining such insight.

[click] As we show next, self-repair can break the conclusion an ablation gives you.

[click] In this toy example, layer 2 plays a key role in shaping the prediction when nothing is ablated.

[click] We ablate it — but layer 3 repairs the damage,

[click] and the output barely changes,

[click] so one concludes that layer 2 does not make a difference.

[click] In reality, layer 3 was compensating for it.

[click] In other words, self-repair leads to a conclusion that looks plausible and is wrong.
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
    <rect class="box" :class="{ on: $clicks === 2 }" x="1265" y="190" width="895" height="1240" />
    <rect class="box" :class="{ on: $clicks === 3 }" x="2225" y="190" width="795" height="1240" />
    <g :class="{ on: $clicks === 1 }">
      <text x="690" y="130">clean run</text>
    </g>
    <g :class="{ on: $clicks === 2 }">
      <text x="1150" y="320">B reacts freely</text>
      <line x1="1480" y1="355" x2="1680" y2="455" marker-end="url(#ar)" />
    </g>
    <g :class="{ on: $clicks === 3 }">
      <text x="2000" y="430">B held clean</text>
      <line x1="2330" y1="465" x2="2400" y2="590" marker-end="url(#ar)" />
    </g>
  </svg>
</div>

<div class="eff-cite">
  Figure 5 in McGrath et al., <em>The Hydra Effect</em>, <a href="https://arxiv.org/abs/2307.15771">arXiv:2307.15771</a>
</div>

<div class="defs">
  <div class="lead"><b>Effect</b> = &Delta;<i> y</i> caused by an intervention</div>
  <ul>
    <li :class="{ on: $clicks >= 2 }"><b>Total effect (TE)</b>: ablate A, let B react freely</li>
    <li :class="{ on: $clicks >= 3 }"><b>Direct effect (DE)</b>: ablate A, freeze B at its clean value</li>
    <li :class="{ on: $clicks >= 4 }"><b>Compensation effect (CE)</b> = DE &minus; TE: how much B compensates for the ablation of A</li>
  </ul>
  <div class="verdict" :class="{ on: $clicks >= 5 }">
    <b>self-repair</b> &nbsp;&hArr;&nbsp; <b>CE &gt; 0</b> <span class="qual">(consistently)</span>
  </div>
</div>

<div class="click-anchors">
  <span v-click></span><span v-click></span><span v-click></span>
  <span v-click></span><span v-click></span>
</div>

<style scoped>
.eff-cite {
  text-align: center; font-size: 0.62rem; opacity: 0.55; margin-top: 0.25rem;
}
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
To answer the research question we need to define self-repair and quantify it. An effect is the change in the output logit caused by an intervention.

We rely on different types of effects.

[click] We first run the clean run, with no intervention, and save the activations for later use.

[click] Next we ablate layer A and let everything downstream react freely. That gives the total effect.
This is what a plain ablation measures.

[click] If instead we hold the downstream at its clean values, only the A-to-y path carries the intervention, and we get the direct effect.

[click] The gap between them is the compensation effect: how much damage downstream layers compensate

[click] So self-repair means a compensation effect that is positive across different inputs.
-->

---

# Why both TE and DE are needed

- A two-layer transformer, layer $A$ and layer $B$
- Layer $A$ constantly writes $a = 1$

<div>

* The intervention is $do(A = 0)$
* The output $y$ is defined as

$$
y = \begin{cases}
1 & \text{if } a + b > 0 \\
0 & \text{otherwise}
\end{cases}
$$

</div>

<div class="toy fade" :class="{ on: $clicks >= 1, s1: $clicks === 2, s2: $clicks === 3, s3: $clicks >= 4 }">
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
  <span v-click></span><span v-click></span>
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
.toy-tbl td:first-child { font-weight: 400; font-style: italic; }
.toy-tbl th.c-te, .toy-tbl td.c-te,
.toy-tbl th.c-de, .toy-tbl td.c-de {
  border-left: 2px dashed transparent; border-right: 2px dashed transparent;
  transition: border-color 0.3s ease;
}
.toy-tbl th.c-te, .toy-tbl th.c-de { border-top: 2px dashed transparent; }
.toy.s1 .toy-tbl tbody tr:nth-child(1) td.c-te,
.toy.s1 .toy-tbl tbody tr:nth-child(2) td.c-te {
  border-left-color: #ef4444; border-right-color: #ef4444;
}
.toy.s1 .toy-tbl tbody tr:nth-child(1) td.c-te { border-top: 2px dashed #ef4444; }
.toy.s1 .toy-tbl tbody tr:nth-child(2) td.c-te { border-bottom: 2px dashed #ef4444 !important; }
.toy.s2 .toy-tbl tbody tr:nth-child(2) td.c-de,
.toy.s2 .toy-tbl tbody tr:nth-child(3) td.c-de {
  border-left-color: #ef4444; border-right-color: #ef4444;
}
.toy.s2 .toy-tbl tbody tr:nth-child(2) td.c-te,
.toy.s2 .toy-tbl tbody tr:nth-child(3) td.c-te { border-right-color: #ef4444; }
.toy.s2 .toy-tbl tbody tr:nth-child(2) td.c-de { border-top: 2px dashed #ef4444; }
.toy.s2 .toy-tbl tbody tr:nth-child(3) td.c-de { border-bottom: 2px dashed #ef4444 !important; }
.toy.s3 .toy-tbl tbody td.c-te { border-left-color: #16a34a; }
.toy.s3 .toy-tbl tbody td.c-de { border-right-color: #16a34a; }
.toy.s3 .toy-tbl tbody tr:first-child td.c-te,
.toy.s3 .toy-tbl tbody tr:first-child td.c-de { border-top: 2px dashed #16a34a; }
.toy.s3 .toy-tbl tbody tr:last-child td.c-te,
.toy.s3 .toy-tbl tbody tr:last-child td.c-de { border-bottom: 2px dashed #16a34a !important; }
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
So why do we need both? Here is a two-layer toy model. Layer A always writes 1 and is the layer we ablate, and the output is one if the two layers' total output is above zero.

[click] There are three scenarios, all with the same model output: B is redundant, it repairs, or A is load-bearing.

[click] Read the total effect alone and redundant and repaired look identical — both are zero.

[click] Read the direct effect alone and repaired and load-bearing look identical — both are one.

[click] Only looking at the two effects together tells all three apart.
-->

---

# What we measured

<div class="did">

<div class="did-row">
  <div class="did-k">Models</div>
  <div class="did-v">LimiX-2M &middot; Mitra &middot; TabICLv2 &middot; TabFM</div>
</div>
<div class="did-row">
  <div class="did-k">Tasks</div>
  <div class="did-v">15 binary classification datasets</div>
</div>
<div class="did-row fade" :class="{ on: $clicks >= 1 }">
  <div class="did-k">Intervention</div>
  <div class="did-v">ablate one layer at a time, every layer</div>
</div>
<div class="did-row fade" :class="{ on: $clicks >= 1 }">
  <div class="did-k">TE</div>
  <div class="did-v">ablate the layer and re-run forward pass</div>
</div>
<div class="did-row fade" :class="{ on: $clicks >= 1 }">
  <div class="did-k">DE</div>
  <div class="did-v">path patching</div>
</div>

</div>

<div class="click-anchors"><span v-click></span></div>

<style scoped>
.click-anchors { font-size: 0; line-height: 0; height: 0; }
.fade { opacity: 0; transition: opacity 0.35s ease; }
.fade.on { opacity: 1; }
.did { width: fit-content; margin: 2rem auto 0; }
.did-row { display: flex; align-items: baseline; margin-bottom: 0.85rem; }
.did-k {
  width: 9rem; flex: none; font-weight: 700; font-size: 1.05rem;
}
.did-v { font-size: 1.05rem; }
</style>

<!--
In our experiment, we check 4 state-of-the-art tabular foundation models, on 15 binary classification datasets.

[click] We ablate one layer at a time, for all the layers.
The total effect is easy: re-run the forward pass and let the downstream react.
The direct effect uses a technique called path patching.
-->

---

# Our main result: self-repair is weak in TFMs

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
Here is our main finding: the evidence for self-repair is weak in TFMs.

For reference, we first show the language-model case, where the evidence is strong. Each dot is a (layer, input) pair, and the x and y axes are DE and TE.

[click] Points below the diagonal are the cases where self-repair happens, because DE exceeds TE and the compensation effect is therefore positive. For the language model under study, this happens consistently.

[click] Now the same measurement on a tabular foundation model.

[click] The region below the diagonal is almost empty, which says self-repair is rare. We show one model here; the others look very similar.
-->

---

# Implications

<div class="means">

<div class="m-row">
  <div class="m-mark good">&#10003;</div>
  <div class="m-txt">Self-repair does not appear to be a serious concern in TFMs &mdash;
  ablation can be used as it is.</div>
</div>

<div class="m-row fade" :class="{ on: $clicks >= 1 }">
  <div class="m-mark warn">!</div>
  <div class="m-txt">Our analysis covers 4 models and 15 datasets.
  A general claim needs more evidence.</div>
</div>

</div>

<div class="click-anchors"><span v-click></span></div>

<style scoped>
.click-anchors { font-size: 0; line-height: 0; height: 0; }
.fade { opacity: 0; transition: opacity 0.35s ease; }
.fade.on { opacity: 1; }
.means { max-width: 46rem; margin: 2rem auto 0; }
.m-row { display: flex; gap: 1rem; margin-bottom: 1.4rem; }
.m-mark {
  flex: none; width: 1.6rem; text-align: center;
  font-size: 1.3rem; font-weight: 700; opacity: 0.55;
}
.m-mark.good { color: #16a34a; opacity: 1; }
.m-mark.warn { color: #ef4444; opacity: 1; }
.m-txt { font-size: 1.05rem; line-height: 1.6; }
</style>

<!--
What does the result mean?

self-repair does not look like a serious concern in tabular foundation models, so ablation can be used as it is — a layer that reads as unimportant is indeed unimportant.

[click] However, our analysis covers four models and fifteen datasets. A general claim needs more evidence.
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
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
}
.link-wrap a {
  font-size: 1.9rem; font-weight: 600;
  color: inherit; text-decoration: none;
}
</style>

<!--
If you want to know more, please check out the full article. Thanks for watching this video.
-->
