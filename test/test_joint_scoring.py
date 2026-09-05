"""Tests for joint (in-context) candidate scoring in the tree search.

The properties that pin the semantics down:

1. `tree_messages` agrees with `get_tree_msg` on every node.
2. `evaluate_tree_branch`, re-propagating only one branch, agrees with a full `evaluate_tree` of the
   modified tree.
3. At depth 0 the tree is empty, so the joint score of a candidate reduces exactly to its isolated
   score - `evaluate_path([candidate, root])`. This is the analogue, for scoring, of the invariant
   that `evaluate_tree` generalizes `evaluate_path` on a single-child chain.
4. Where branches share a real component ancestor the two scores diverge substantially, because the
   messages merge at that component's `ln2` + nonlinearity. (If they agreed everywhere the flag
   would be pointless.)
5. Scoring is *simultaneous*: two candidates under the same leaf are each scored against a context
   containing neither, so an already-admitted duplicate cannot suppress its twin. This is what keeps
   redundant/backup components in the circuit, which completeness needs.

Everything here is non-positional (`position=None`), as the searches are actually run for IOI. It is
not a stylistic choice: pinning a name mover to the final token leaves it nothing to read, and the
resulting contributions land at ~1e-5, which is the float32 floor of this metric - every ratio then
comes out as a rounded quantization artefact rather than a measurement.
"""

from functools import partial

import pytest
import torch
import transformer_lens

from ipe.graph_search import _joint_scoring_context, _score_candidate
from ipe.metrics import target_logit_percentage
from ipe.nodes import ATTN_Node, EMBED_Node, FINAL_Node, MLP_Node
from ipe.paths import evaluate_path, evaluate_tree, get_tree_msg, tree_messages, evaluate_tree_branch


############################################### Fixtures #####################################################

@pytest.fixture(scope="module")
def model():
    model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")
    model.eval()
    return model


@pytest.fixture(scope="module")
def metric(model):
    """Target-logit metric on a single IOI prompt, in the plain (zero-patching) configuration."""
    _, cache = model.run_with_cache(
        ["When Mary and John went to the shop, John gave a drink to"], prepend_bos=True)
    cache = dict(cache)
    return cache, partial(
        target_logit_percentage,
        clean_final_resid=cache[f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"],
        model=model,
        target_tokens=[model.to_single_token(" Mary")],
    )


############################################### Helpers ######################################################

def _kw(cache):
    return dict(msg_cache=cache, cf_cache={}, patch_type="zero")


def attn(model, cache, layer, head):
    return ATTN_Node(model, layer=layer, head=head, position=None, keyvalue_position=None, **_kw(cache))


def mlp(model, cache, layer):
    return MLP_Node(model, layer=layer, position=None, **_kw(cache))


def embed(model, cache):
    return EMBED_Node(model, position=None, **_kw(cache))


def make_root(model, cache, metric_fn):
    return FINAL_Node(model, layer=model.cfg.n_layers - 1, position=None, metric=metric_fn,
                      **_kw(cache))


def attach(parent, child):
    """Wire `child` under `parent` the way the search does."""
    child.parent = parent
    parent.children.add(child)
    return child


############################################### Tests ########################################################

def test_tree_messages_matches_get_tree_msg(model, metric):
    """The cached messages are the same ones the plain recursion computes."""
    cache, metric_fn = metric

    root = make_root(model, cache, metric_fn)
    head = attach(root, attn(model, cache, 9, 9))
    attach(head, mlp(model, cache, 0))
    attach(head, embed(model, cache))

    messages = tree_messages(root)
    assert torch.allclose(messages[id(root)], get_tree_msg(root), atol=1e-4)
    for node in (head, *head.children):
        assert id(node) in messages


def test_evaluate_tree_branch_matches_full_reevaluation(model, metric):
    """Re-propagating one branch equals rebuilding and re-scoring the whole tree."""
    cache, metric_fn = metric

    root = make_root(model, cache, metric_fn)
    leaf = attach(root, attn(model, cache, 9, 9))
    attach(root, mlp(model, cache, 10))          # a sibling subtree to merge with at the root

    messages = tree_messages(root)
    candidate = embed(model, cache)
    branch = leaf.forward(message=candidate.forward(message=None))
    incremental = evaluate_tree_branch(root, leaf, branch, metric_fn, messages)

    attach(leaf, candidate)                      # now actually build it and score the whole thing
    full = evaluate_tree(root, metric_fn)
    assert float(incremental) == pytest.approx(float(full), abs=1e-3)


def test_joint_equals_isolated_at_depth_zero(model, metric):
    """With an empty tree there is no context, so the joint score is the isolated one."""
    cache, metric_fn = metric

    root = make_root(model, cache, metric_fn)
    joint = _joint_scoring_context(root, [root], metric_fn)

    for candidate in (mlp(model, cache, 11), attn(model, cache, 9, 9)):
        candidate.parent = root
        isolated = evaluate_path([candidate, root], metric_fn)
        scored = _score_candidate(candidate, root, [root], root, metric_fn, joint)
        assert float(scored) == pytest.approx(float(isolated), abs=1e-3)


@pytest.mark.parametrize("shared_layer, leaf_head, context", [
    (11, (9, 9), [(10, 7)]),
    (10, (9, 9), [(9, 6), (9, 0)]),
    (9, (8, 6), [(7, 3), (7, 9)]),
])
def test_joint_differs_from_isolated_under_a_shared_ancestor(model, metric, shared_layer,
                                                             leaf_head, context):
    """Branches sharing a real component diverge: their messages merge at its `ln2` + nonlinearity.

    Measured relative gaps on this prompt are 8-44% for the three cases below, against ~2-5% when
    the branches meet only at the root (where `FINAL_Node.forward` is the identity, so the sole
    nonlinearity is `ln_final` inside the metric). The 1% bar is well under the former and well over
    float32 noise, which is ~1e-5 relative on contributions of this size."""
    cache, metric_fn = metric

    root = make_root(model, cache, metric_fn)
    shared = attach(root, mlp(model, cache, shared_layer))
    leaf = attach(shared, attn(model, cache, *leaf_head))
    for layer, head in context:
        attach(shared, attn(model, cache, layer, head))

    joint = _joint_scoring_context(root, [leaf], metric_fn)
    suffix = [leaf, shared, root]
    candidate = embed(model, cache)
    candidate.parent = leaf

    isolated = float(evaluate_path([candidate] + suffix, metric_fn))
    scored = float(_score_candidate(candidate, leaf, suffix, root, metric_fn, joint))
    assert abs(scored - isolated) > 1e-2 * abs(isolated), (
        f"joint scoring collapsed onto isolated scoring ({scored:+.5f} vs {isolated:+.5f}); "
        "the context is not being applied"
    )


def test_scoring_is_simultaneous(model, metric):
    """A candidate's score does not depend on what else was admitted at the same depth.

    Two candidates under one leaf are both measured against a context that contains neither, so a
    redundant twin is not suppressed by its partner - which is what keeps compensating components
    (the IOI backup name movers) in the circuit."""
    cache, metric_fn = metric

    root = make_root(model, cache, metric_fn)
    shared = attach(root, mlp(model, cache, 10))
    leaf = attach(shared, attn(model, cache, 9, 9))
    suffix = [leaf, shared, root]

    first, second = mlp(model, cache, 0), mlp(model, cache, 1)
    first.parent = second.parent = leaf

    joint = _joint_scoring_context(root, [leaf], metric_fn)
    before = float(_score_candidate(second, leaf, suffix, root, metric_fn, joint))

    # Admit `first`, as the search does mid-depth, and re-score `second` against the SAME frozen
    # context. The score must not move: the context was captured before either was attached.
    attach(leaf, first)
    after = float(_score_candidate(second, leaf, suffix, root, metric_fn, joint))
    assert after == pytest.approx(before, abs=1e-5)
