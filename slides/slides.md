---
theme: default
title: Do tabular foundation models repair themselves?
info: TFM self-repair — 2-minute talk
class: text-center
transition: slide-left
mdc: true
---

# Do tabular foundation models repair themselves?

Han Xiao

---

# Background

<div class="lead-line">Language models <em>self-repair</em>: ablate one layer, later layers compensate for the damage</div>

<figure class="hydra">
  <svg class="hyd" viewBox="0 0 940 430">
    <line class="ax" x1="80" y1="350" x2="885" y2="350" />
    <line class="ax" x1="80" y1="30" x2="80" y2="350" />
    <line class="tick" x1="90" y1="350" x2="90" y2="358" />
    <text class="ticklab" x="90" y="378" text-anchor="middle">1</text>
    <line class="tick" x1="176" y1="350" x2="176" y2="358" />
    <text class="ticklab" x="176" y="378" text-anchor="middle">2</text>
    <line class="tick" x1="261" y1="350" x2="261" y2="358" />
    <text class="ticklab" x="261" y="378" text-anchor="middle">3</text>
    <line class="tick" x1="347" y1="350" x2="347" y2="358" />
    <text class="ticklab" x="347" y="378" text-anchor="middle">4</text>
    <line class="tick" x1="432" y1="350" x2="432" y2="358" />
    <text class="ticklab" x="432" y="378" text-anchor="middle">5</text>
    <line class="tick" x1="518" y1="350" x2="518" y2="358" />
    <text class="ticklab" x="518" y="378" text-anchor="middle">6</text>
    <line class="tick" x1="604" y1="350" x2="604" y2="358" />
    <text class="ticklab" x="604" y="378" text-anchor="middle">7</text>
    <line class="tick" x1="689" y1="350" x2="689" y2="358" />
    <text class="ticklab" x="689" y="378" text-anchor="middle">8</text>
    <line class="tick" x1="775" y1="350" x2="775" y2="358" />
    <text class="ticklab" x="775" y="378" text-anchor="middle">9</text>
    <line class="tick" x1="860" y1="350" x2="860" y2="358" />
    <text class="ticklab" x="860" y="378" text-anchor="middle">10</text>
    <line class="vguide" :class="{ on: $clicks >= 1 }" x1="432" y1="30" x2="432" y2="350" />
    <text class="axlab" x="480" y="412" text-anchor="middle">layer</text>
    <text class="axlab" x="30" y="190" text-anchor="middle" transform="rotate(-90 30 190)">effect on logits</text>
    <g class="legend">
      <line class="clean" x1="686" y1="292" x2="731" y2="292" />
      <text class="leg" x="741" y="298">pre-ablation</text>
      <line class="abl on" x1="686" y1="326" x2="731" y2="326" />
      <text class="leg" x="741" y="332">post-ablation</text>
    </g>
    <polyline class="clean" points="90,255 176,242 261,247 347,234 432,126 518,212 604,199 689,204 775,186 860,191" />
    <polyline class="abl" :class="{ on: $clicks >= 1 }" points="347,234 432,298" />
    <polyline class="abl" :class="{ on: $clicks >= 2 }" points="432,298 518,66 604,186 689,195 775,178 860,189" />
    <g class="ann" :class="{ on: $clicks >= 1 }">
      <text x="418" y="340" text-anchor="end">ablate this layer</text>
    </g>
    <g class="ann" :class="{ on: $clicks >= 2 }">
      <text x="534" y="70">a later layer repairs</text>
    </g>
  </svg>
  <figcaption>
    Illustration, after McGrath et al., <em>The Hydra Effect</em>, <a href="https://arxiv.org/abs/2307.15771">arXiv:2307.15771</a>, Fig. 1.
  </figcaption>
</figure>

<div class="rq"><b>Research question</b>: do tabular foundation models repair themselves as well?</div>

<div class="click-anchors"><span v-click></span><span v-click></span></div>

<style scoped>
.click-anchors { font-size: 0; line-height: 0; height: 0; }
.hydra { margin: 0.6rem 0; }
.lead-line { font-size: 1.05rem; }
.rq { text-align: center; font-size: 1.5rem; margin-top: 1rem; }
.hyd { width: 78%; display: block; margin: 0 auto; }
.hyd .ax { stroke: #9ca3af; stroke-width: 2; }
.hyd .vguide {
  stroke: #9ca3af; stroke-width: 2; stroke-dasharray: 7 6;
  opacity: 0; transition: opacity 0.3s ease;
}
.hyd .vguide.on { opacity: 1; }
.hyd .clean { fill: none; stroke: #2563eb; stroke-width: 4; }
.hyd .abl {
  fill: none; stroke: #ef4444; stroke-width: 4;
  opacity: 0; transition: opacity 0.4s ease;
}
.hyd .abl.on { opacity: 1; }
.hyd .axlab { fill: #6b7280; font-size: 22px; }
.hyd .ann { opacity: 0; transition: opacity 0.3s ease; }
.hyd .ann.on { opacity: 1; }
.hyd .ann text { fill: #111; font-size: 24px; font-weight: 700; }
.hyd .leg { fill: #6b7280; font-size: 19px; }
.hyd .tick { stroke: #9ca3af; stroke-width: 2; }
.hyd .ticklab { fill: #9ca3af; font-size: 18px; }
.hyd .legend line { stroke-width: 4; }
.hydra figcaption {
  font-size: 0.7rem; opacity: 0.65; text-align: center; margin-top: 0.3rem;
}
</style>


---

# What is a tabular foundation model (TFM)?

- TFMs $\approx$ transformers for tabular data

<div class="tfm-fig">

<div class="col">
  <div class="fade on">
    <div class="cap">context table</div>
    <table class="tbl">
      <tr><td>x<sub>11</sub></td><td>x<sub>12</sub></td><td class="y">y<sub>1</sub></td></tr>
      <tr><td>x<sub>21</sub></td><td>x<sub>22</sub></td><td class="y">y<sub>2</sub></td></tr>
      <tr><td>⋯</td><td>⋯</td><td class="y">⋯</td></tr>
      <tr><td>x<sub>n1</sub></td><td>x<sub>n2</sub></td><td class="y">y<sub>n</sub></td></tr>
    </table>
  </div>
  <div class="fade" :class="{ on: $clicks >= 1 }">
    <div class="cap cap-lo">test table</div>
    <table class="tbl">
      <tr><td>x<sub>11</sub></td><td>x<sub>12</sub></td><td class="y q">?</td></tr>
      <tr><td>⋯</td><td>⋯</td><td class="y q">⋯</td></tr>
      <tr><td>x<sub>k1</sub></td><td>x<sub>k2</sub></td><td class="y q">?</td></tr>
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

<div class="arrow fade" :class="{ on: $clicks >= 3 }">&rarr;</div>

<div class="col fade" :class="{ on: $clicks >= 3 }">
  <div class="cap">prediction</div>
  <table class="tbl">
    <tr><td class="y hat">ŷ<sub>1</sub></td></tr>
    <tr><td class="y hat">⋯</td></tr>
    <tr><td class="y hat">ŷ<sub>k</sub></td></tr>
  </table>
</div>

</div>

<div class="fade" :class="{ on: $clicks >= 4 }">

- TFMs do *in-context learning*, i.e., there is **no** gradient update

</div>

<div class="click-anchors">
  <span v-click></span><span v-click></span>
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

<div class="click-anchors"><span v-click></span><span v-click></span></div>

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

---
clicks: 6
---

# Why the RQ matters for AI safety

- TFMs are deployed in high-stake domains, e.g., lending: $y = 1$ grant the loan, $y = 0$ deny
- Ablation is a main tool for understanding how a model works &mdash; and self-repair breaks it

<div class="sf">

<div class="col">
  <div class="cap">context table</div>
  <table class="tbl">
    <tr><td>x<sub>11</sub></td><td>x<sub>12</sub></td><td class="y">1</td></tr>
    <tr><td>x<sub>21</sub></td><td>x<sub>22</sub></td><td class="y">0</td></tr>
    <tr><td>⋯</td><td>⋯</td><td class="y">⋯</td></tr>
    <tr><td>x<sub>n1</sub></td><td>x<sub>n2</sub></td><td class="y">1</td></tr>
  </table>
  <div class="cap cap-lo">test table</div>
  <table class="tbl">
    <tr><td>x<sub>11</sub></td><td>x<sub>12</sub></td><td class="y q">?</td></tr>
    <tr><td>x<sub>21</sub></td><td>x<sub>22</sub></td><td class="y q">?</td></tr>
  </table>
</div>

<div class="arrow">&rarr;</div>

<div class="model">
  <div class="model-title">Transformer</div>
  <div class="blk">layer 1</div>
  <div class="blk" :class="$clicks >= 2 ? 'abl' : ($clicks >= 1 ? 'imp blinking' : 'imp')">layer 2</div>
  <div class="blk" :class="{ comp: $clicks >= 2, blinking: $clicks >= 5 }">layer 3</div>
  <div class="blk">layer 4</div>
</div>

<div class="arrow">&rarr;</div>

<div class="col">
  <div class="cap">prediction</div>
  <div class="pred-box" :class="{ flash: $clicks >= 3 }">
    <table class="tbl">
      <tr><td class="hat">ŷ<sub>1</sub> = 1</td></tr>
      <tr><td class="hat">ŷ<sub>2</sub> = 0</td></tr>
    </table>
  </div>
  <div class="same" :class="{ flash: $clicks >= 3 }">unchanged</div>
</div>

</div>

<div class="verdict-wrap">
  <div class="verdict" :class="{ shown: $clicks >= 4, struck: $clicks >= 6 }">
    &ldquo;Layer 2 does not contribute.&rdquo;
  </div>
</div>

<div class="click-anchors">
  <span v-click></span><span v-click></span><span v-click></span>
  <span v-click></span><span v-click></span><span v-click></span>
</div>

<style scoped>
.click-anchors { font-size: 0; line-height: 0; height: 0; }
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
    <b>self-repair</b> &nbsp;&hArr;&nbsp; <b>CE &gt; 0</b>
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
.click-anchors { font-size: 0; line-height: 0; height: 0; }
</style>

---

# Why both TE and DE are needed

- A two-layer transformer, layer $A$ and layer $B$
- Layer $A$ constantly writes $a = 1$
- The intervention is $do(A = 0)$
- The output $y$ is defined as

$$
y = \begin{cases}
1 & \text{if } a + b > 0 \\
0 & \text{otherwise}
\end{cases}
$$

<div class="toy" :class="{ s1: $clicks === 1, s2: $clicks >= 2 }">
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
    </tbody>
  </table>
  <div class="ann ann1">self-repair = redundancy &nbsp;&#10060;</div>
  <div class="ann ann2">self-repair &ne; redundancy &nbsp;&#9989;</div>
</div>

<div class="click-anchors"><span v-click></span><span v-click></span></div>

<style scoped>
.click-anchors { font-size: 0; line-height: 0; height: 0; }
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
.toy-tbl th { font-weight: 600; }
.toy-tbl th.c-te, .toy-tbl td.c-te,
.toy-tbl th.c-de, .toy-tbl td.c-de {
  border-left: 2px dashed transparent; border-right: 2px dashed transparent;
  transition: border-color 0.3s ease;
}
.toy-tbl th.c-te, .toy-tbl th.c-de { border-top: 2px dashed transparent; }
.toy.s1 th.c-te, .toy.s1 td.c-te { border-left-color: #9ca3af; border-right-color: #9ca3af; }
.toy.s1 th.c-te { border-top-color: #9ca3af; }
.toy.s1 tbody tr:last-child td.c-te { border-bottom: 2px dashed #9ca3af; }
.toy.s2 th.c-te, .toy.s2 td.c-te { border-left-color: #9ca3af; }
.toy.s2 th.c-de, .toy.s2 td.c-de { border-right-color: #9ca3af; }
.toy.s2 th.c-te, .toy.s2 th.c-de { border-top-color: #9ca3af; }
.toy.s2 tbody tr:last-child td.c-te,
.toy.s2 tbody tr:last-child td.c-de { border-bottom: 2px dashed #9ca3af; }
.ann {
  position: absolute; top: 100%; margin-top: 0.4rem;
  font-size: 1.15rem; font-weight: 600;
  white-space: nowrap; opacity: 0; transition: opacity 0.3s ease;
}
.ann1 { color: #ef4444; }
.ann2 { color: #16a34a; }
.ann1 { left: 62%; }
.ann2 { left: 68%; }
.toy.s1 .ann1 { opacity: 1; }
.toy.s2 .ann2 { opacity: 1; }
</style>


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
