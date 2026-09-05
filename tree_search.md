# The Tree Extension of IPE

This document is the reference description of the tree-based circuit search
(`TreeMessagePatching`): the formalism it operates in, the two scoring rules it supports, the
properties each one has, and how the resulting circuits are evaluated. It is written to be read on
its own; the accompanying code is `src/ipe/paths.py`, `src/ipe/graph_search.py`, the comparison
driver `experiments/tree_vs_path.py`, and the evaluation notebook
`experiments/faithfulness_completeness.ipynb`.

Appendix A records the development history, including two design decisions that were later found to
be wrong and the corrections that replaced them.

---

## 1. Motivation

`PathMessagePatching` returns a circuit as a *set of independent root-to-embedding paths*
`[EMBED, …, FINAL]`, each scored by its own isolated effect. Many of those paths share long suffixes
toward the root — hundreds may end in `… → A9H9 → FINAL` — and the path search re-derives each such
suffix from scratch. Two questions follow:

1. **Representation.** Can the search grow a *single tree rooted at FINAL* that shares those suffixes,
   turning a redundant path set into a trie with one explicit parent per node?
2. **Objective.** Once a tree exists, can a candidate be scored *in the context of the branches
   already admitted* rather than in isolation — and does that find a different circuit?

The two questions are separable, and the implementation keeps them separate: the tree is always a
trie, and the scoring rule is a flag (`joint_scoring`). Section 4.1 shows that under isolated scoring
the answer to (2) is "no by construction" — the tree is then exactly the path set, re-materialised.

---

## 2. Preliminaries: message passing over the computational graph

### 2.1 Nodes are difference operators

Every `Node` implements `forward` as a map from *a perturbation at the component's input* to *the
perturbation it induces at the component's output*, not as a function of activations. For an MLP:

```python
residual = msg_cache[input_name] - message          # perturb the input
residual = ln2(residual)
return  msg_cache[output_name] - mlp.forward(residual)   # the resulting change at the output
```

Write this operator `Δ_out = f_v(Δ_in)`. Called with `message=None` a node emits its own signal:
`out_clean − out_cf` under counterfactual patching, `out_clean` under zero patching.

### 2.2 Paths

`evaluate_path([n_k, …, n_1, root])` composes the operators down the chain and applies the metric to
the final residual with the accumulated message removed:

```
m_k = f_{n_k}(None),   m_{i} = f_{n_i}(m_{i+1}),   score = metric( root.forward() − m_root )
```

### 2.3 Trees

`get_tree_msg` generalises this by **summing the children's messages at a node's input and pushing
them through `f_v` once, jointly**:

```python
incoming = sum(get_tree_msg(child) for child in node.children)
return node.forward(message=incoming)
```

`evaluate_tree(root, metric) = metric(root.forward() − get_tree_msg(root))`, written `𝔽(T)` below. A
childless node emits `forward(None)`, so a single-child chain reduces exactly to `evaluate_path` —
an invariant `experiments/tree_vs_path.py::check_invariant` asserts at the start of every run, using
the run's own caches and patch configuration.

**A node plays one of two roles depending on where it sits, and that distinction carries most of the
semantics of the method:**

| position | message | meaning |
|---|---|---|
| **leaf** | `f_v(None)` | the node is *being ablated*. Its message is its own clean-minus-counterfactual output, read from the caches — **frozen**, and unaffected by anything else in the tree. |
| **internal** | `f_v(incoming)` | the node is *reacting*. Its message is how its own output changes when its children's contributions are removed from its input. |

An ablation's downstream consequences are therefore represented **only through internal nodes**. This
is what makes `evaluate_tree` a *path-restricted* ablation — path patching, and the point of the
method — rather than a full knockout in which every component in the model reacts at once. Section 4.6
measures what that restriction does and does not cost.

### 2.4 The non-additivity everything rests on

`f_v` is **not linear**, for three independent reasons: LayerNorm (`ln1`/`ln2`) rescales by the norm
of the *perturbed* residual; the MLP is nonlinear; and an attention head whose query or key stream is
patched recomputes its pattern. Therefore, for two children `a, b` of a node `v`:

$$ f_v(m_a + m_b) \;\neq\; f_v(m_a) + f_v(m_b). $$

Every difference between the two scoring rules of Section 3 is a consequence of this inequality. Note
that the *root* is an exception: `FINAL_Node.forward` returns its message unchanged, so branches that
meet only at the output interact solely through the LayerNorm inside the metric. Section 4.3 shows
this makes their interaction small, and that the real interaction happens at shared *internal* nodes.

### 2.5 Two boundary facts

**Empty tree scores zero.** `evaluate_tree` on a childless root uses `message = 0` (not
`forward(None)` — see Appendix A.2), so `𝔽(∅) = metric(root.forward())`. The metric is calibrated so
that the unperturbed residual scores zero: `ExperimentManager.load_metric` computes `baseline_value`
from precisely the cache the root reads back. Hence **`𝔽(∅) = 0` exactly**.

**A node's children saturate at the node itself.** Under counterfactual patching, if `S` is the set
of *all* of `L`'s predecessors then `Σ_{c∈S} m_c = in_L^{clean} − in_L^{cf}`, so

$$ f_L\Big(\sum_{c \in S} m_c\Big) = out^{clean}_L - L\big(in^{cf}_L\big) = out^{clean}_L - out^{cf}_L = f_L(\text{None}). $$

Growing the tree beneath `L` therefore *interpolates* from "ablate `L` wholesale" (no children) down
to "ablate only what the admitted children explain". This is the fact that makes the naive marginal
definition ill-posed (Section 3.3).

---

## 3. The search

### 3.1 Backwards breadth-first growth

`TreeMessagePatching` starts from the root and grows the tree level by level. At each depth every
frontier leaf `L` enumerates its predecessors (`get_expansion_candidates`), each candidate `C` is
scored, and the admitted ones are attached as children of `L`. Non-embedding admissions form the next
frontier. A leaf whose candidates all fail simply stops growing, and survives as a truncated branch.

### 3.2 Scoring rule A — isolated branch contribution (`joint_scoring=False`, default)

$$ \mathrm{score}(C) \;=\; \texttt{evaluate\_path}\big([C, L, \ldots, \mathrm{root}]\big) $$

The candidate's message travels to the root alone; every intermediate node sees only that one
message. The score depends on the chain and nothing else — not on siblings, not on any other branch,
not on the threshold. This is exactly the per-path score `PathMessagePatching` uses.

### 3.3 Scoring rule B — joint, in-context contribution (`joint_scoring=True`)

$$ \mathrm{score}(C) \;=\; \mathbb{F}\big(T \text{ with } L \text{ emitting } f_L(m_C)\big) \;-\; \mathbb{F}\big(T \text{ with } L \text{ emitting } 0\big), $$

with `m_C = f_C(None)` and every other branch of `T` frozen. The candidate's message now merges with
the other subtrees' messages at every shared ancestor and at the metric.

**On the choice of baseline.** The natural-looking definition `𝔽(T ⊕ C) − 𝔽(T)` — the discrete
derivative of the tree's score with respect to attaching one edge — is *wrong here*, and was the
original v1 rule (Appendix A.1). By the saturation fact of Section 2.5, attaching a child to `L`
does not add signal to the tree: it *narrows* an ablation, replacing `f_L(None)` by the strictly
smaller `f_L(m_C)`. That difference is positive only at depth 0 (where attaching to the empty root
genuinely adds mass) and **systematically negative afterwards**, so it is not comparable across
depths and cannot drive a single threshold.

Taking "`L` contributes nothing" as the baseline instead fixes this. The resulting score keeps the
isolated rule's sign convention and scale, and because `𝔽(∅) = 0` it reduces **exactly** to
`evaluate_path([C, root])` at depth 0, where the tree is still empty. The two rules therefore agree
on the first level and diverge only once there is a context to be in.

### 3.4 Simultaneity within a depth

Joint scores depend on what is already in `T`, so the BFS must say what `T` is while a depth is being
scored. Three options:

| policy | order-independent | penalises redundancy | cost |
|---|---|---|---|
| **simultaneous** (implemented) | yes | no, *by design* | 1× |
| sequential (attach as you go) | no | yes | 1× |
| greedy re-rank (attach best, re-score, repeat) | yes | yes | k× |

The implementation **freezes `T` for the whole depth** (`_joint_scoring_context`): every candidate at
a depth is scored against a context containing none of its same-depth peers, and admitted nodes are
attached against that same frozen snapshot.

This is a deliberate choice against redundancy pruning. Completeness (Section 5.2) is *defined* as the
absence of components outside the circuit that compensate for removals from it — on IOI, precisely
the backup name movers. A greedy re-ranking rule would reject a backup head as soon as the head it
backs up is admitted, buying minimality at the direct cost of the property being measured. Under
simultaneous scoring two candidates carrying the same signal are both admitted, which is the wanted
behaviour. (The order-independent way to get a redundancy penalty is to average marginals over
orderings — a Shapley value — which is not affordable at this scale.)

What joint scoring still buys, then, is **context across depths**: a candidate whose effect is
already carried by a branch admitted at a shallower depth scores lower, and one that matters only in
the presence of such a branch scores higher.

Section 4.5 shows that this residual cross-depth suppression is not harmless: it is what makes the
joint tree drop the early MLP block on IOI, at a cost of −0.72 normalised faithfulness. Simultaneity
protects redundancy *within* a depth but cannot protect it *across* depths, and on a serial stack like
GPT-2's MLPs that is where the damage happens.

### 3.5 Admission rules

Both admission strategies exist for both searches, so the comparison can be run under either budget:

| strategy | path search | tree search | rule |
|---|---|---|---|
| `threshold` | `PathMessagePatching` | `TreeMessagePatching` | admit any candidate with contribution ≥ `min_contribution` (or `|contribution|` ≥ it, with `include_negative`) |
| `topk` | `PathMessagePatching_LimitedLevelWidth` | `TreeMessagePatching_LimitedLevelWidth` | at each depth score all candidate extensions of all leaves, keep the global top `max_width` by `|contribution|` |

`include_negative` is unchanged by the scoring rule, because rule B preserves rule A's sign
convention. Its reading under joint scoring is *"admit `C` if routing through `L` changes the tree's
score by at least `t`, in either direction"*, which still admits negative-effect components such as
the IOI negative name movers. Setting `include_negative=False` turns joint scoring into literal
greedy maximisation of `𝔽` and drops them.

`topk` compares the two searches at a matched *width* budget rather than a matched threshold, which
matters because the threshold means slightly different things when path counts differ.

### 3.6 Complexity

Naively, one joint score costs a full `𝔽` over the tree — `O(|T|)` forward calls, fatal once `T` has
thousands of nodes. It is not necessary: attaching a candidate under `L` invalidates the messages
only along the chain `L → root`, since a change under `L` cannot reach the root by any other route
and the siblings it meets at each ancestor are unaffected. Two helpers exploit this:

- `tree_messages(root)` — caches the message every node emits, keyed by **object identity**
  (`Node.__hash__` hashes by component identity, so the same head in two branches would collide);
- `evaluate_tree_branch(root, node, message, metric, messages)` — re-propagates only the
  `node → root` chain, reusing the cached sibling messages at each level.

Because `T` is frozen for a whole depth (Section 3.4) the cache is built **once per depth** and never
invalidated during scoring. One joint score therefore costs `O(depth)` forward calls — the same order
as `evaluate_path`. Measured on GPT-2 small / IOI (threshold 0.5, 3 prompts): **7.9 s joint vs 8.0 s
isolated** for the whole search.

---

## 4. Properties

### 4.1 Under isolated scoring the tree *is* the path set

With rule A a candidate's score depends only on its chain, so the tree admits a chain exactly when
the path search admits the corresponding path prefix. The tree is then the same set of paths
materialised as a suffix-sharing trie, and any path pruning strategy transfers directly to it. This
is what makes the path-vs-tree comparison well-posed: differences in the recovered circuits are
attributable to the representation, not to a different objective.

One asymmetry survives and is intrinsic to the algorithms rather than to the scoring: the path search
only *returns* paths that reached the embeddings, discarding frontier nodes that never completed a
path, whereas the tree keeps every admitted node. At a matched threshold the tree's circuit is
therefore systematically the larger of the two, and comparisons should be read at a matched *budget*.

That is a statement about *node sets*. The trie also has a capability the path set does not, and
Section 4.6 measures it: because compensation is carried by **internal** nodes (Section 2.3), a
structure recording only paths-to-output cannot express "A's effect, as modified by B" — B has to be
*on* the path, above A. Self-repair is representable in a trie and not in a set of independent
root-to-embedding paths. This is the strongest argument for the tree representation, and it holds
under either scoring rule.

### 4.2 Threshold monotonicity, and when a sweep is derivable from one run

Under **isolated** scoring, two facts hold: scores are threshold-independent (rule A never reads
`min_contribution`), and admission is downward-closed (a chain admitted at `t'` is admitted at any
`t < t'`, and candidate generation does not depend on the threshold). Therefore

$$ \{\text{chains at } t'\} \;=\; \{\text{chains at } t_{\text{base}} : \min |\text{contribution}| \geq t'\}, $$

so a *single* run at the lowest threshold reproduces the run at every higher threshold **exactly**, by
pruning. This was verified empirically: searches run at `t_base = 0.3` and pruned to `0.5 / 1.0 / 2.0`
produce node sets, edge sets and branch counts identical to searches executed directly at those three
thresholds, for both the path and the tree search.

Under **joint** scoring this fails. A score is only valid for the tree that existed when it was
taken, and that tree depends on the threshold: lowering it admits more shallow branches, which changes
the context every deeper candidate is scored against. A threshold sweep under joint scoring therefore
requires **one full search per threshold**. The extra cost is milder than it looks — the lowest
threshold dominates and higher ones are progressively cheaper (measured: 7 s / 4 s / 2 s at
0.5 / 1.0 / 2.0) — so a whole sweep costs roughly twice a single run at the base threshold.

### 4.3 Where the two rules actually diverge

Measured on one IOI prompt with `target_logit_percentage`, non-positionally, as the relative gap
between the joint and isolated score of the same candidate:

| branches merge at… | relative gap |
|---|---|
| the root only | **2–5 %** |
| a shared component ancestor (an MLP) | **8–44 %** |

The pattern follows directly from Section 2.4: `FINAL_Node.forward` is the identity, so branches
meeting only at the output interact solely through `ln_final` inside the metric, which is nearly
linear over the relevant range. The substantial interaction happens at a shared component's
`ln2` + nonlinearity. **Joint scoring therefore matters precisely where the trie shares internal
nodes** — that is, exactly where the tree representation is doing something a path set does not.

> **Measurement caveat.** These numbers must be taken non-positionally. Pinning a name mover to the
> final token leaves it nothing to read, and contributions then land at ~1e-5, the float32 floor of
> this metric; every ratio computed there is a rounded quantisation artefact rather than a
> measurement. `test/test_joint_scoring.py` documents this.

### 4.4 Faithfulness is not monotone in circuit size

It is natural to expect that keeping more components makes a circuit more faithful — the circuit is
"closer to the model", so it should behave more like it. **This is false**, and on IOI it fails
dramatically enough to invert the trend of a whole threshold sweep.

The sweep below is the tree search under joint scoring, node granularity, scope `all` (everything
outside the circuit ablated). Configuration: gpt2-small, IOI, `logit_difference` with counterfactual
denoising, `batch=5`, `target_length=15`, non-positional, `include_negative=True`,
`base_min_contribution=0.05`; evaluated by ABC mean-ablation knockout over 64 held-out IOI prompts.
`F(M) = +3.169`, `F(∅) = +0.058`, normaliser `3.111`.

| threshold | comps | heads | MLPs | branches | `F(C)` | faithfulness |
|---|---|---|---|---|---|---|
| 0.050 | 41 | 33 | 8 | 392 | −1.245 | **−0.419** |
| 0.085 | 29 | 24 | 5 | 217 | −0.887 | −0.304 |
| 0.143 | 22 | 17 | 5 | 147 | −0.694 | −0.242 |
| 0.243 | 17 | 15 | 2 | 84 | +0.111 | +0.017 |
| 0.412 | 13 | 13 | 0 | 48 | +0.105 | +0.015 |
| 0.697 | 11 | 11 | 0 | 27 | +0.098 | +0.013 |
| 1.181 | 7 | 7 | 0 | 16 | +0.070 | +0.004 |
| 2.000 | 5 | 5 | 0 | 11 | +0.067 | +0.003 |

Faithfulness *rises* as the circuit shrinks from 41 components to 5, and the largest circuit is the
only one that scores **below the empty circuit**. The path search on the same axis does not invert —
0.828 at `t=0.05` (27 comps) down to 0.012 at `t=0.143` (13 comps) — so this is not the harness.

**The mechanism is a partially ablated serial stack.** Holding the tree's 33 heads at `t=0.05` fixed
and varying only which MLPs are kept:

| circuit (same 33 heads throughout) | MLPs kept | `F(C)` | faithfulness |
|---|---|---|---|
| tree @0.05 as found | 0, 5, 6, 7, 8, 9, 10, 11 | −1.245 | **−0.419** |
| tree @0.05, m0 only | 0 | −0.072 | −0.042 |
| tree @0.05, no MLPs at all | — | +0.106 | **+0.016** |
| tree @0.05 + m1, m2, m3, m4 | all 12 | +1.005 | **+0.304** |

Keeping **eight** MLPs scores far worse than keeping **none**. Ablation replaces a component's output
with its ABC-mean, which is a roughly neutral value; keeping m5–m11 while m1–m4 are ablated instead
lets the late MLPs compute on a *corrupted* early residual and propagate the corruption forward. A
broken prefix of a serial stack is worse than no stack. The rising trend in the sweep is therefore not
"smaller is more faithful" — it is the harmful partial-MLP configuration disappearing as the tree
sheds its MLPs entirely (8 → 5 → 5 → 2 → 0 → 0).

The same four MLPs move the path circuit by almost exactly as much, in the opposite direction:

| circuit | MLPs kept | `F(C)` | faithfulness |
|---|---|---|---|
| path @0.05 as found (20 heads) | 0, 1, 2, 3, 4, 9, 10 | +2.633 | **+0.828** |
| path @0.05 − m1, m2, m3, m4 | 0, 9, 10 | −0.020 | **−0.025** |

Adding four components swings the tree by **+0.72**; removing the same four swings the path by
**−0.85**. Everything else — heads, edges, thresholds, evaluation set — is held fixed. The early MLP
block is load-bearing for IOI under this knockout, which is consistent with the standard observation
that MLP0 in GPT-2 small acts as an extension of the token embedding, with m1–m4 continuing to build
the name representation the attention circuit then reads.

**How to read the tables in light of this.** Faithfulness is a property of the *ablated model*, not a
monotone score over subsets, so a faithfulness curve that falls with circuit size is not by itself
evidence of a bug — but it is always worth asking which components entered or left. Two consequences:
a faithfulness ranking is only meaningful between circuits that are not differently broken; and a
circuit can be penalised far more for a missing *bridge* component than rewarded for many correct
ones.

### 4.5 Joint scoring drops components the isolated rule keeps

Section 4.4 leaves a question: why did the tree miss m1–m4 when the path search found them? It is not
the tree representation, and it is not reachability. It is the scoring rule. At `t = 0.05`, same
configuration, same batch:

| tree run at `t = 0.05` | components | heads | MLPs found |
|---|---|---|---|
| `joint_scoring=False` (isolated) | 48 | 36 | 0,1,2,3,4,5,6,7,8,9,10,11 — **all 12** |
| `joint_scoring=True` (joint) | 41 | 33 | 0, 5,6,7,8,9,10,11 — **m1–m4 dropped** |

**They were scored, not missed.** Of the eight parents under which the isolated run admitted m1–m4
(`a3.h0`, `a5.h5`, `a6.h9`, `a8.h6`, `m3`, `m4`, `m5`, `m7`), **six are present in the joint tree**, so
m1–m4 were generated as candidates there and rejected on their score.

**Why they were rejected is a threshold crossing.** Compare m0, the one early MLP that survives, across
the two runs — same component, same tree, only the rule differs:

| m0 | occurrences | depths | max \|contribution\| |
|---|---|---|---|
| isolated | 107 | 2–8 | **0.797** |
| joint | 18 | 2, 4 | **0.168** |

Joint scoring rescaled m0 by 4.7×, and m0 survived only because it had ~16× headroom over the `0.05`
threshold. In the isolated run m1–m4 peak at **0.113 / 0.123 / 0.110 / 0.110** — barely 2.2× above the
bar. A comparable rescaling puts them under it, and they vanish entirely. Nothing about the rule
"rejects redundant components"; it rescales, and the marginal ones fall through.

**Measured, and the answer is two answers.** Section 12 of
`experiments/faithfulness_completeness.ipynb` recovers the rejected scores directly: it rebuilds the
joint tree from its cache as live `Node`s, truncates it to depth `k` (which reproduces the frozen
context the BFS saw when it expanded that depth, exactly rather than approximately), and re-scores
m1–m4 under every placement the joint tree offered, with both rules. The failures split:

| candidate | best isolated *in the joint tree* | best joint | best in the isolated *run* | |
|---|---|---|---|---|
| m1 | **0.073** (under `a5.h5`) | 0.010 | 0.113 | threshold crossing |
| m3 | **0.074** (under `a8.h6`) | 0.032 | 0.110 | threshold crossing |
| m2 | 0.010 | 0.015 | 0.123 | **structural** |
| m4 | 0.019 | 0.019 | 0.110 | **structural** |

For m1 and m3 the account above holds: the placement existed and cleared `0.05` under the isolated
rule but not the joint one. For **m2 and m4 it does not** — the joint tree never offered a placement
either rule would have admitted, although the isolated run admitted them at 0.123 and 0.110. They were
lost with the branches that carried them, further up the tree.

That distinction matters more than the original finding. The two rules do not merely disagree
candidate-by-candidate; they grow **structurally different trees**, and the disagreement compounds with
depth, because a different shallow structure means different chains and therefore different scores for
the same component under the same parent *component*. It also sets the bar for a fix: a repair applied
at the point of rejection is not enough.

**What the rescaling factor is has not been established, and it is not redundancy.** Two measurements,
scoring candidates both ways in one frozen context:

*Distribution* — over the 14 candidates of a real frontier whose isolated score clears `0.25 × t`
(anything smaller is fp32 noise; see the caveat in Section 4.3), the ratio |isolated| / |joint| runs
from **0.46 to 3.02**, median 0.88 — a 6.6× spread, so not a uniform rescaling. But ratios *below* 1
are common: joint scoring **amplifies** about as often as it attenuates.

*Controlled test* — one candidate (`m0` under `a5.h5` under `a9.h9`), varying only which sibling sits
beside the leaf, so the two messages merge at `a9.h9`'s input:

| sibling | sibling size | ratio |
|---|---|---|
| *none* (control on the machinery) | — | **1.00** |
| **`m0` — a literal duplicate** | 0.044 | **2.53** |
| `m10` — unrelated | 0.286 | **2.66** |
| `a5.h8` — unrelated | 0.031 | 1.92 |
| `a8.h6` — unrelated | 3.383 | 1.91 |
| `a7.h9` — unrelated | 1.192 | **0.97** |
| `a11.h2` — unrelated | 0.021 | 0.93 |

An unrelated MLP attenuates the candidate *more* than its own literal duplicate does, and a sibling 27×
the duplicate's size attenuates it not at all. **There is no separation between duplicate and control**,
so the joint/isolated ratio is not a redundancy measure and must not be reported as one. Note also that
both branches in that test are *leaves* — neither reacts (Section 2.3) — so the test sits in the regime
where redundancy cannot show up at all. Section 4.6 covers the regime where it does.

Whatever the mechanism, the consequence for the metric is not in doubt:

> **"Adds little in context" is not the same as "can be ablated harmlessly."**
> Joint scoring admits on the first; ablation-based faithfulness measures the second. The two came
> apart here by −0.72 of normalised faithfulness, against the ≈0.1 head-F1 that joint scoring costs on
> ground-truth overlap.

**Consequence.** For the headline path-vs-tree tables, run with `joint_scoring=False`. That is the
well-posed comparison anyway (Section 4.1: matched scoring, so differences are attributable to the
representation), it restores the derivable threshold sweep (Section 4.2), and it avoids this artefact.
Joint scoring belongs in the chapter as a **negative result** with the diagnosis above — which is
stronger evidence for the isolated design than the head-overlap numbers of Section 5.3.

### 4.6 What a path-restricted ablation can and cannot see

`evaluate_tree` **is** an ablation — it removes a component's contribution and propagates the
consequence through the real nonlinearities of every node the perturbation passes through. What it
restricts is *where* the consequence may travel: along the tree's own edges, and nowhere else. A
component the tree does not contain keeps its clean output no matter what happens upstream.

The practical question is whether that restriction hides **self-repair** — a component compensating for
another's removal, which is the phenomenon completeness is defined around. It does not, provided the
edge is there. Two experiments on the IOI name movers `A = {a9.h9, a9.h6, a10.h0}` and their backups
`B = {a10.h10, a10.h6, a10.h2, a10.h1, a11.h2, a9.h7, a9.h0, a11.h9}`, gpt2-small, batch 5,
non-positional, `logit_difference` with counterfactual denoising.

**Flat topology — every component a leaf, so nothing reacts:**

| tree | `𝔽` |
|---|---|
| `A` only | +9.526 |
| `B` only | +1.689 |
| `A + B`, all as direct children of FINAL | +11.174 |
| sum of the separate scores | +11.215 |
| **deviation** | **−0.041** (0.4%) |

Additive to within `ln_final`'s contribution (Section 4.3's 2–5% for branches meeting only at the
root). The marginal of `B` given `A` is +1.648 against +1.689 alone — ×0.98. No trace of redundancy.

**Routed topology — the backups as internal nodes above the name movers:**

| tree | `𝔽` |
|---|---|
| `A` as direct children of FINAL (direct path only) | +9.526 |
| `A` direct, **plus** `FINAL ← backup_i ← A` for all 14 layer-legal routing edges | **+6.993** |
| **difference** | **−2.533 (−27%)** |

The routed branches carry an opposite-signed message: accounting for the fact that removing the name
movers *changes what the backups output* cancels 27% of the damage. **That is self-repair, measured
inside the message formalism.**

The two results together give the rule:

> A leaf is being ablated and is frozen; an internal node is reacting. Compensation is captured
> **exactly along the edges the tree contains**, and not at all for components that are only ever
> leaves. A flat set of root-attached components cannot express it; a trie can.

This also explains why the flat comparison showed no redundancy signature where a full knockout shows a
large one. Under a knockout (`F`, everything recomputes) the same two sets give:

| | drop `A` | drop `B` | drop both | sum | super-additivity |
|---|---|---|---|---|---|
| `F`, full knockout | 0.137 | 0.181 | **1.570** | 0.318 | **+1.252** |

with the marginal of `B` given `A` at **1.433** against 0.181 alone — ×7.9. The knockout's "`B` alone"
is small *because* `A` compensates; the flat tree's "`B` alone" is large because `A` is frozen. The
tree is not blind to the phenomenon — the flat topology simply measures a different quantity.

**The search does find these edges.** Both real trees at `t = 0.05` contain **8** name-mover → backup
routing edges (`a9.h9→a10.h10`, `a9.h9→a11.h2`, `a9.h6→a10.h6`, `a9.h6→a11.h2`, …) out of 229 unique
edges (115 head→head) for the isolated run and 177 (98 head→head) for the joint one. Whether a circuit's
incompleteness tracks how much of this routing it captured is an obvious thing to test and has not been.


---

## 5. Evaluation

### 5.1 Structural comparison — `experiments/tree_vs_path.py`

One driver runs both searches with identical settings (same model, batch, metric and caches — each
search gets its own `clone_root` of one configured `FINAL_Node`) and reports the components
discovered, the runtime, and the overlap. Flags: `--task {ioi,greater-than}`,
`--strategy {threshold,topk}`, `--min-contribution`, `--max-width`, `--positional`, `--batch-size`,
`--metric`, `--joint-scoring`.

**IOI.** Setup mirrors `experiments/MIB/run_search.py`: `mib-bench/ioi` prompts with the
`s2_io_flip_counterfactual`, counterfactual (denoising) patching, and the `run_search` convention of
feeding counterfactual prompts as the clean run and vice-versa; batches are bucketed to a single
tokenised length. Ground truth is the Wang et al. (2022) 26-head set grouped by functional role; the
report marks each known head `[PT]`/`[P-]`/…, lists off-circuit heads and the MLP layers found (IOI
has no canonical MLP ground truth, so these are descriptive only), and draws the tree.

**Greater-than.** Ground truth here is a full circuit *graph*, not a head set, so `GREATER_THAN_EDGES`
encodes the published data-flow edges and `build_ground_truth_tree` parses them into the same `Node`
structure the search produces. The report compares tree against tree: branches split into common /
ground-truth-only / tree-only / incomplete, plus ASCII drawings and, for positional runs, a
layer × token-position DAG grid.

### 5.2 Causal comparison — `experiments/faithfulness_completeness.ipynb`

Structural overlap with a head set says nothing about whether a circuit *does the work*. The notebook
evaluates both searches by **knockout**, in the sense of Wang et al. (2023, §3): given a circuit `C`,
run the model with everything outside `C` mean-ablated over the ABC distribution and read the IOI
logit difference `F(C)`.

- **Faithfulness** — `F(C) ≈ F(M)`; reported normalised so `1.0` is the full model and `0.0` the empty
  circuit.
- **Completeness** — for every `K ⊆ C`, `F(C\K) ≈ F(M\K)`; estimated over randomly sampled `K`, mean
  and max, normalised the same way.

The evaluation is deliberately independent of the discovery signal: a *different* intervention (ABC
mean-ablation, not counterfactual denoising) on a *different, larger* prompt set. Circuits are read
off at two granularities — components, and `(source → destination, stream)` edges with
stream ∈ `q/k/v/mlp/resid` — with the edge stream recovered from the destination's own patch flags.
Wang et al.'s circuit, the full model, the empty circuit and size-matched random circuits are scored
alongside as references, each recomputed by the same harness rather than quoted.

> **The ground-truth baseline keeps the MLPs, and must.** Wang et al. specify their circuit as a set
> of *attention heads*; the MLPs are outside the specification, not outside the circuit. Applying the
> knockout rule literally — ablate everything not in `C` — deletes all 12 MLP sublayers, and that
> alone destroys the model: keeping **all 144 attention heads** with the MLPs ablated scores a
> faithfulness of **−0.19**, below the empty circuit. Under that reading every head set scores ≈ 0
> regardless of quality (26 heads: 0.03), which silently flatters anything compared against it. With
> the MLPs kept the same 26 heads score **1.55**. Both rows are reported; the MLP-keeping one is the
> target. The comparison is still not like-for-like, in the other direction: the ground-truth circuit
> is handed its MLPs while the searches must discover theirs, and no choice of convention fixes that.

Because of Section 4.2 the notebook's sweep is asymmetric: the path search runs once and is pruned
across thresholds, while the tree search runs once per threshold whenever `CFG.tree_joint_scoring` is
on. Setting it to `False` restores the single-run behaviour.

### 5.3 Recorded results

> Report files (`ioi_circuit_report.txt`, `greater_than_report.txt`) are overwritten by every run, so
> each result below is stated with the configuration that produced it.

**IOI head recovery** (gpt2-small, `topk`, `max_width=20`, batch 4, positional, `logit_difference`,
isolated scoring): path and tree recover the **same 17/26 known heads** — all 3 Name Movers, both
Negative Name Movers, 4/8 Backup Name Movers, 3/4 S-Inhibition, 2/4 Induction, 3/3 Duplicate Token,
0/2 Previous Token. Path produced 161 complete paths / 70 unique nodes; the tree 90 branches / 65
unique nodes. This is the expected outcome under matched scoring (Section 4.1).

**Joint vs isolated scoring — structural** (gpt2-small, `threshold=0.5`, batch 3, non-positional,
`logit_difference`) — a single coarse run:

| scoring | nodes | branches | ground-truth heads |
|---|---|---|---|
| isolated | 49 | 35 | **13** |
| joint | 43 | 34 | **11** |

Joint scoring dropped the induction heads `(5,5)` and `(6,9)`, whose effect is already carried by the
S-inhibition branches above them — the rule working as designed, at a small cost in ground-truth
recall. Both settings kept the negative name movers.

**Joint vs isolated scoring — causal** (gpt2-small, `base_min_contribution=0.05`, batch 5, eval on 64
held-out prompts, node granularity, scope `all`): the head-overlap cost above is the small part. Joint
scoring also drops the early MLP block m1–m4, which costs **−0.72 normalised faithfulness** at
`t = 0.05` (−0.419 as found, +0.304 with m1–m4 restored). Section 4.5 has the diagnosis and Section 4.4
the mechanism. **Joint scoring is worse on both axes measured so far**; the isolated rule is the one to
use for the headline tables.

**Greater-than** (Llama-3.2-1B-Instruct, `topk`, `max_width=10`, batch 20, positional): the ground
truth expands to 812 branches; the tree search found 15 complete branches (26 incomplete) with **0**
in common. Branch-level recall is very low at this width budget, and the caveats below apply.

**Known caveats of the greater-than setup** (against the official `gpt2-greater-than` repo), in order
of importance:

1. **Metric mismatch.** The official metric is the probability difference
   `Σ P(YY' > YY) − Σ P(YY' ≤ YY)` over all valid two-digit year tokens; the experiment uses a
   single-token `logit_difference` proxy (`yy+1` good vs `yy−1` bad). The published circuit was
   discovered under prob-diff, so the proxy weakens the comparison.
2. **Effectively unbalanced batch.** Candidates are generated noun-outer / YY-inner and bucketed by
   token length, so the default batch is one noun with the smallest YYs. The official dataset
   balances YY uniformly over 2–98 across 120 nouns.
3. **Century fixed at 17**; the official pool spans centuries 10–18 filtered through
   `get_valid_years`.
4. A hand-picked 14-noun list instead of the official `cache/potential_nouns.txt`.

---

## 6. Granularity and limitations

- Both searches expand attention at the **per-head** level
  (`get_expansion_candidates(..., include_head=True)`); MLPs are per-block.
- The path search's `batch_heads` two-stage block-then-head shortcut is **not** implemented for the
  tree, so runtime comparisons should disable it (`batch_heads=False`) for parity.
- Both searches are **greedy threshold-gated BFS**: a strong deep node sitting behind a weak
  intermediate node is never reached, at any threshold. A threshold sweep faithfully reproduces what
  the algorithm finds at each threshold; it does not find the best circuit of a given size.
- Joint scoring is defined against the whole frozen tree. A cheaper siblings-only variant (accounting
  for interaction at `L`'s input but not at the ancestors above it) is *not* implemented; it would be
  an approximation of the implemented rule, not an alternative semantics.
- Joint scoring suppresses serially redundant components across depths and should not be used for
  faithfulness tables (Sections 4.4–4.5). It remains available as a documented negative result.
- Faithfulness under knockout is **not monotone in circuit size** (Section 4.4), so circuits should not
  be ranked by it without checking that they are not differently broken.

---

## 7. Implementation map

| file | role |
|---|---|
| `src/ipe/paths.py` | `evaluate_path`; `get_tree_msg` / `evaluate_tree` (joint ablation); `tree_messages` and `evaluate_tree_branch` (the per-depth message cache and `O(depth)` re-propagation) |
| `src/ipe/graph_search.py` | `TreeMessagePatching` (threshold), `TreeMessagePatching_LimitedLevelWidth` (top-k beam), both with `joint_scoring`; `_joint_scoring_context` and `_score_candidate`; the path-search counterparts; `setup_tree_debug_log` |
| `experiments/tree_vs_path.py` | driver: batch loading (IOI / greater-than), invariant check, both searches, overlap stats, ground-truth reports, ASCII tree and position-grid rendering; `--joint-scoring` |
| `experiments/faithfulness_completeness.ipynb` | knockout harness (node and edge granularity), faithfulness / completeness sweep, comparison tables and LaTeX export |
| `test/test_joint_scoring.py` | the cache/re-propagation identities, the depth-0 reduction, the divergence under a shared ancestor, and simultaneity |
| `ioi_circuit_report.txt`, `greater_than_report.txt` | latest generated reports (overwritten per run) |
| `tree_search_debug.log` | per-depth trace of the last tree run |

---

## Appendix A: Development history

### A.1 v1 — marginal contribution against a joint-ablation baseline (`8c556f7`, 2026-06-15)

The first implementation scored each candidate by
`score(C) = evaluate_tree(T + C) − evaluate_tree(T)`, with the baseline frozen per BFS level so that
scoring was order-independent within a depth. The stated motivation was that the raw tree score is
dominated by already-discovered branches and cannot discriminate an individual candidate.

**This rule is ill-posed past depth 0**, for the reason given in Section 3.3: attaching a child
narrows an ablation instead of adding one, so the difference is positive at depth 0 and systematically
negative afterwards. The observation that "the metric grows monotonically with the ablated mass" holds
only for attachments *to the root*; the direction reverses for every deeper attachment. The current
joint rule (Section 3.3) replaces the baseline with "`L` contributes nothing", which is comparable
across depths.

### A.2 The empty-tree baseline bug (`0e8f934`, 2026-06-16)

`evaluate_tree` on a childless root called `get_tree_msg(root)` = `root.forward(None)`, so the
baseline for the empty tree ablated the *entire* final residual instead of nothing. Fixed to
`message = get_tree_msg(root) if root.children else 0`. This is what makes `𝔽(∅) = 0` (Section 2.5),
which in turn is what makes the joint rule reduce to the isolated one at depth 0.

### A.3 v1.5 — sibling-message caching (`16af7c6`, 2026-06-16)

Scoring one candidate by re-evaluating the whole tree is `O(|T|)` forwards. `refresh_tree_messages`
and `candidate_message` cached each node's outgoing and summed-incoming messages and recomputed only
the `leaf → root` chain. The idea was sound and survives: `tree_messages` /
`evaluate_tree_branch` (Section 3.6) are its current form.

### A.4 v2 — isolated branch contribution (`6a3e1d6`, 2026-06-17)

The marginal rule was dropped entirely in favour of the isolated branch contribution, on the grounds
that a candidate's admission should not depend on which sibling subtrees happened to be discovered
first, and that the path search scores every path in isolation — so under the marginal rule the two
searches were optimising different objectives and their differences could not be attributed to the
tree structure. The sibling caching was removed from the scoring loop, and `evaluate_tree` was
retained for *reporting* the joint ablation of the finished tree.

### A.5 Current — both rules, behind a flag

Joint scoring was reinstated as `joint_scoring`, defaulting to off, with the corrected baseline
(A.1), the simultaneous within-depth policy of Section 3.4, and the caching of A.3 restored as
`tree_messages` / `evaluate_tree_branch`. Both objectives are now reachable from one code path:
`joint_scoring=False` keeps the well-posed path-vs-tree comparison of A.4 and the derivable threshold
sweep of Section 4.2; `joint_scoring=True` scores in context at the cost of one search per threshold.

---

## Appendix B: Open questions

- **Does joint scoring help?** ~~Open~~ — answered, negatively, on IOI: it costs ~0.1 head-F1
  (Section 5.3) and −0.72 normalised faithfulness (Sections 4.4–4.5), because it prunes serially
  redundant components that ablation-based faithfulness requires to be present. What remains open is
  whether the failure is specific to serial stacks like GPT-2's early MLPs, or general. A cheap test:
  re-run the sweep on a task whose circuit has no comparable bridge component and see whether the two
  rules converge.
- **Why exactly were m1–m4 rejected?** ~~Open~~ — answered in Section 4.5 by the re-scoring diagnostic
  (notebook §12): m1 and m3 are threshold crossings, m2 and m4 are structural losses further up the
  tree. What remains open is whether the structural divergence is generic or particular to this run.
- **Does captured routing predict completeness?** Section 4.6 shows compensation is represented exactly
  along the edges the tree holds, and that the search finds 8 such edges on IOI. If a circuit's
  incompleteness falls as it captures more of the compensating routing, that is both an explanation of
  the completeness score and a lever for improving it.
- **Union admission.** Admitting on *either* rule has a guarantee the structural failures of
  Section 4.5 demand: at depth 0 the two rules coincide (Section 3.3), and if the union tree contains
  the isolated tree at depth `k` then every isolated-admitted placement at `k+1` exists and scores
  identically (the isolated score is context-free), so it is admitted too. By induction **the isolated
  tree is always a subgraph of the union tree**, so no component the isolated rule finds can be lost —
  including the m2/m4 cases a post-hoc repair cannot reach. Both scores are `O(depth)`, so the cost is
  2×. Note this guarantees containment, not a better score: faithfulness is not monotone in circuit
  size (Section 4.4), so the improvement has to be measured.
- **Greater-than fidelity.** Implement the prob-diff metric over the valid-year token indices and
  balance the batch across YY and nouns before length-bucketing, so the ground-truth comparison is
  faithful to Hanna et al. (2023).
- **Runtime parity.** Port the `batch_heads` block-then-head expansion shortcut to the tree search, so
  path-vs-tree runtime is measured at equal expansion cost.
- **Redundancy without losing completeness.** Greedy re-ranking prunes redundant siblings but destroys
  the property completeness measures (Section 3.4). Whether a rule exists that reports redundancy
  without suppressing it — for instance admitting redundant siblings but *labelling* them as mutually
  substitutable — is open.
- **Other ground truths.** IOI is currently the only task with a head-level ground truth in use; ACDC
  is not yet vendored under `benchmark/Automatic-Circuit-Discovery`, and the notebook has a section
  ready for it.
