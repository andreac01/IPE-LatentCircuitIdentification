from functools import partial
import torch
import transformer_lens
import pytest
from ipe.nodes import ATTN_Node, EMBED_Node, FINAL_Node, MLP_Node
from typing import Any

############################################### Utility hooks #####################################################

def single_head_hook(z_scores: torch.Tensor, hook: Any, head: int) -> torch.Tensor:
    """
    single_head_hook is a hook function that isolates a specific attention head by setting all other heads to zero.
    """
    z_scores[:, :, :head, :] = torch.zeros_like(z_scores[:, :, :head, :])
    z_scores[:, :, head + 1:, :] = torch.zeros_like(z_scores[:, :, head + 1:, :])
    return z_scores

def single_value_pos_hook(value_resid: torch.Tensor, hook: Any, keyvalue_position: int) -> torch.Tensor:
    """
    single_value_pos_hook is a hook function that isolates the contribution from a specific value position by setting all other positions to zero.
    """
    value = torch.zeros_like(value_resid)
    value[:, keyvalue_position, :] = value_resid[:, keyvalue_position, :].detach().clone()
    return value

def plot_hook(z: torch.Tensor, hook: Any) -> torch.Tensor:
    """
    plot_hook is a hook function that plots the attention scores.
    """
    import matplotlib.pyplot as plt
    if z.dim() == 4:
        z = torch.mean(z, dim=0)
    for i in range(z.shape[0]):
        
        plt.imshow(z[0].detach().cpu().numpy(), cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.xlabel('Key/Value Position')
        plt.ylabel('Query Position')
        plt.show()
    return z

################################################ Utility functions #####################################################

def simple_logit_diff(model, clean_residual, corrupted_resid, target_token_id):
    """
    simple_logit_diff computes the logit difference for a single target token given clean and corrupted residuals.
    """
    clean_logits = model.unembed(model.ln_final(clean_residual))  # Shape: (batch_size, seq_len, vocab_size)
    corrupted_logits = model.unembed(model.ln_final(corrupted_resid))  # Shape: (batch_size, seq_len, vocab_size)

    # Get the logits for the target token
    clean_target_logits = clean_logits[:, -1, target_token_id]  # Shape: (batch_size,)
    corrupted_target_logits = corrupted_logits[:, -1, target_token_id]  # Shape: (batch_size,)

    # Compute logit difference
    logit_diff = clean_target_logits - corrupted_target_logits  # Shape: (batch_size,)

    return logit_diff.mean()  # Return average logit difference over batch


def get_cache_bwd(model, prompt, completion, metric):
    """
    get_cache_bwd computes the gradients of the metric with respect to the model's activations using backpropagation.
    """
    model.reset_hooks()
    model.requires_grad_(True)
    model.zero_grad()
    cache = {}
    grad_cache = {}

    hooks = ["hook_embed"]
    hooks += [f"blocks.{i}.hook_resid_pre" for i in range(model.cfg.n_layers)]
    hooks += [f"blocks.{i}.hook_attn_out" for i in range(model.cfg.n_layers)]
    hooks += [f"blocks.{i}.hook_resid_mid" for i in range(model.cfg.n_layers)]
    hooks += [f"blocks.{i}.hook_mlp_out" for i in range(model.cfg.n_layers)]
    hooks += [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]
    def forward_cache_hook(residual, hook):
        cache[hook.name] = residual
    def backward_cache_hook(residual, hook):
        grad_cache[hook.name] = residual
    
    with torch.enable_grad():
        for hook in hooks:
            model.add_hook(hook, forward_cache_hook, "fwd")
            model.add_hook(hook, backward_cache_hook, "bwd")
    
        model.run_with_hooks(prompt, prepend_bos=True)
        target_token_id = model.to_single_token(completion)

        final_residual = cache[f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"]
        corrupted_resid = final_residual

        value = metric(model, final_residual.detach().clone(), corrupted_resid, target_token_id)
        value.backward()
    model.reset_hooks()
    return dict(grad_cache)
    

################################################ Fixtures and Tests #####################################################

@pytest.fixture(scope="module")
def gpt2_model():
    model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")
    return model

# Note: Test also with qwen to ensure compatibility with grouped query attention
@pytest.fixture(scope="module")
def qwen_model():
    model = transformer_lens.HookedTransformer.from_pretrained("Qwen/Qwen2.5-0.5B")
    return model


@pytest.fixture(scope="module")
def clean_prompt():
    return "Hello world"


@pytest.fixture(scope="module")
def corrupted_prompt():
    return "Hi John"


@pytest.fixture(scope="module")
def gpt2_msg_cache(gpt2_model, clean_prompt):
    _, cache = gpt2_model.run_with_cache(clean_prompt, prepend_bos=True)
    assert cache['hook_embed'].shape[1] == 3, f"The prompt should be tokenized into 3 tokens. Got {cache['hook_embed'].shape[1]} for prompt '{clean_prompt}'"
    return dict(cache)

@pytest.fixture(scope="module")
def qwen_msg_cache(qwen_model, clean_prompt):
    _, cache = qwen_model.run_with_cache(clean_prompt, prepend_bos=True)
    assert cache['hook_embed'].shape[1] == 3, f"The prompt should be tokenized into 3 tokens. Got {cache['hook_embed'].shape[1]} for prompt '{clean_prompt}'"
    return dict(cache)

@pytest.fixture(scope="module")
def gpt2_cf_cache(gpt2_model, corrupted_prompt):
    _, cache = gpt2_model.run_with_cache(corrupted_prompt, prepend_bos=True)
    assert cache['hook_embed'].shape[1] == 3, f"The prompt should be tokenized into 3 tokens. Got {cache['hook_embed'].shape[1]} for prompt '{corrupted_prompt}'"
    return dict(cache)

@pytest.fixture(scope="module")
def qwen_cf_cache(qwen_model, corrupted_prompt):
    _, cache = qwen_model.run_with_cache(corrupted_prompt, prepend_bos=True)
    assert cache['hook_embed'].shape[1] == 3, f"The prompt should be tokenized into 3 tokens. Got {cache['hook_embed'].shape[1]} for prompt '{corrupted_prompt}'"
    return dict(cache)

@pytest.fixture(scope="module")
def gpt2_grad_cache(gpt2_model):
    return get_cache_bwd(gpt2_model, "Hello world", "!", simple_logit_diff)

@pytest.fixture(scope="module")
def qwen_grad_cache(qwen_model):
    return get_cache_bwd(qwen_model, "Hello world", "!", simple_logit_diff)

@pytest.mark.parametrize("model_name", ["gpt2", "qwen"])
@pytest.mark.parametrize("node_class", [MLP_Node, ATTN_Node, EMBED_Node, FINAL_Node])
@pytest.mark.parametrize("patch_type", ["zero", "counterfactual"])
@pytest.mark.parametrize("position", [None, 2])
def test_forward_none_message(node_class, patch_type, model_name, gpt2_model, qwen_model, gpt2_msg_cache, qwen_msg_cache, gpt2_cf_cache, qwen_cf_cache, position):
    if model_name == "gpt2":
        model = gpt2_model
        msg_cache = gpt2_msg_cache
        cf_cache = gpt2_cf_cache
    elif model_name == "qwen":
        model = qwen_model
        msg_cache = qwen_msg_cache
        cf_cache = qwen_cf_cache
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    if node_class == MLP_Node:
        node = MLP_Node(model, layer=1, position=position, msg_cache=msg_cache, cf_cache=cf_cache, patch_type=patch_type)

        if patch_type == "zero":
            expected = torch.zeros_like(msg_cache[f"blocks.1.hook_mlp_out"])
            if position is None:
                expected = msg_cache[f"blocks.1.hook_mlp_out"].detach().clone()
            else:
                expected[:,position,:] = msg_cache[f"blocks.1.hook_mlp_out"][:, position, :].detach().clone()

            if not torch.allclose(node.forward(None), expected, atol=1e-5):
                print(f"MLP_Node forward with zero patch failed. Expected\n{expected}, got \n\n{node.forward(None)}")
                raise AssertionError(f"MLP_Node forward with zero patch failed. Expected {expected}, got {node.forward(None)}")
        
        if patch_type == "counterfactual":
            expected = torch.zeros_like(cf_cache[f"blocks.1.hook_mlp_out"])
            if position is None:
                expected = msg_cache[f"blocks.1.hook_mlp_out"].detach().clone() - cf_cache[f"blocks.1.hook_mlp_out"].detach().clone()
            else:
                expected[:,position,:] = msg_cache[f"blocks.1.hook_mlp_out"][:, position, :].detach().clone() - cf_cache[f"blocks.1.hook_mlp_out"][:, position, :].detach().clone()

            assert torch.allclose(node.forward(None), expected, atol=1e-5), f"MLP_Node forward with counterfactual patch failed. Expected {expected}, got {node.forward(None)}"

            null_cache = {k: torch.zeros_like(v) for k, v in cf_cache.items()}
            node.cf_cache = null_cache
            expected = torch.zeros_like(null_cache[f"blocks.1.hook_mlp_out"])
            if position is None:
                expected = msg_cache[f"blocks.1.hook_mlp_out"].detach().clone()
            else:
                expected[:,position,:] = msg_cache[f"blocks.1.hook_mlp_out"][:, position, :].detach().clone()
            assert torch.allclose(node.forward(None), expected, atol=1e-5), f"MLP_Node forward with zeroed counterfactual cache failed. Expected {expected}, got {node.forward(None)}"
            node.cf_cache = cf_cache  # Restore original cache
    
    elif node_class == ATTN_Node:
        for head in [None, 0]:
            for keyvalue_position in [None, 1]:
                node = ATTN_Node(model, layer=1, head=head, position=position, msg_cache=msg_cache, cf_cache=cf_cache, keyvalue_position=keyvalue_position, patch_type=patch_type)

                if head is None and keyvalue_position is None:
                        if patch_type == "zero":
                            expected = torch.zeros_like(msg_cache[f"blocks.1.hook_attn_out"])
                            if position is None:
                                expected = msg_cache[f"blocks.1.hook_attn_out"].detach().clone()
                            else:
                                expected[:,position,:] = msg_cache[f"blocks.1.hook_attn_out"][:, position, :].detach().clone()

                            assert torch.allclose(node.forward(None), expected, atol=1e-5), f"ATTN_Node forward with zero patch failed. Expected {expected}, got {node.forward(None)}"
                        
                        if patch_type == "counterfactual":
                            expected = torch.zeros_like(cf_cache[f"blocks.1.hook_attn_out"])
                            if position is None:
                                expected = msg_cache[f"blocks.1.hook_attn_out"].detach().clone() - cf_cache[f"blocks.1.hook_attn_out"].detach().clone()
                            else:
                                expected[:,position,:] = msg_cache[f"blocks.1.hook_attn_out"][:, position, :].detach().clone() - cf_cache[f"blocks.1.hook_attn_out"][:, position, :].detach().clone()

                            assert torch.allclose(node.forward(None), expected, atol=1e-5), f"ATTN_Node forward with counterfactual patch failed. Expected {expected}, got {node.forward(None)}"

                else:
                    if head is not None:
                        model.add_hook(f"blocks.1.attn.hook_z", lambda z, hook: single_head_hook(z, hook, head=head))
                    if keyvalue_position is not None:
                        model.add_hook(f"blocks.1.attn.hook_v", lambda value_resid, hook: single_value_pos_hook(value_resid, hook, keyvalue_position=keyvalue_position))

                    clean_msg = model.run_with_cache("Hello world", prepend_bos=True)[1][f"blocks.1.hook_attn_out"]
                    if head is not None:
                        clean_msg -= model.b_O[1]
                    if patch_type == "counterfactual":
                        corrupted_msg = model.run_with_cache("Hi John", prepend_bos=True)[1][f"blocks.1.hook_attn_out"]
                        if head is not None:
                            corrupted_msg -= model.b_O[1]
                    model.reset_hooks()
                    if patch_type == "zero":
                        expected = torch.zeros_like(clean_msg)
                        if position is None:
                            expected = clean_msg.detach().clone()
                        else:
                            expected[:,position,:] = clean_msg[:, position, :].detach().clone()

                        assert torch.allclose(node.forward(None), expected, atol=1e-5), f"ATTN_Node forward with zero patch (head={head}, kv_pos={keyvalue_position}) failed. Expected {expected}, got {node.forward(None)}"
                    if patch_type == "counterfactual":
                        expected = torch.zeros_like(corrupted_msg)
                        if position is None:
                            expected = clean_msg.detach().clone() - corrupted_msg.detach().clone()
                        else:
                            expected[:,position,:] = clean_msg[:, position, :].detach().clone() - corrupted_msg[:, position, :].detach().clone()
                        assert torch.allclose(node.forward(None), expected, atol=1e-5), f"ATTN_Node forward with counterfactual patch (head={head}, kv_pos={keyvalue_position}) failed. Expected {expected}, got {node.forward(None)}"


    elif node_class == EMBED_Node:
        node = EMBED_Node(model, layer=0, position=position, msg_cache=msg_cache, cf_cache=cf_cache)
        if position is None:
            expected = msg_cache["hook_embed"].detach().clone()
        else:
            expected = torch.zeros_like(msg_cache["hook_embed"])
            expected[:,position,:] = msg_cache["hook_embed"][:, position, :].detach().clone()
        assert torch.allclose(node.forward(None), expected, atol=1e-5), f"EMBED_Node forward failed. Expected {expected}, got {node.forward(None)}"

        
    
    elif node_class == FINAL_Node:
        node = FINAL_Node(model, layer=model.cfg.n_layers-1, position=position, msg_cache=msg_cache, cf_cache=cf_cache)
        if position is None:
            expected = msg_cache[f"blocks.{model.cfg.n_layers-1}.hook_resid_post"].detach().clone()
        else:
            expected = torch.zeros_like(msg_cache[f"blocks.{model.cfg.n_layers-1}.hook_resid_post"])
            expected[:,position,:] = msg_cache[f"blocks.{model.cfg.n_layers-1}.hook_resid_post"][:, position, :].detach().clone()
        assert torch.allclose(node.forward(None), expected, atol=1e-5), f"FINAL_Node forward failed. Expected {expected}, got {node.forward(None)}"

    else: 
        raise AssertionError("Test did not run any assertions.")



@pytest.mark.parametrize("model_name", ["gpt2", "qwen"])
@pytest.mark.parametrize("node_class", [MLP_Node, ATTN_Node, EMBED_Node, FINAL_Node])
@pytest.mark.parametrize("position", [None, 2])
def test_forward_with_message(node_class, model_name, gpt2_model, qwen_model, gpt2_msg_cache, qwen_msg_cache, gpt2_cf_cache, qwen_cf_cache, position):
    if model_name == "gpt2":
        model = gpt2_model
        msg_cache = gpt2_msg_cache
        cf_cache = gpt2_cf_cache
    elif model_name == "qwen":
        model = qwen_model
        msg_cache = qwen_msg_cache
        cf_cache = qwen_cf_cache
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if node_class == MLP_Node:
        node = MLP_Node(model, layer=1, position=position, msg_cache=msg_cache, cf_cache=cf_cache, patch_type="zero")
        node_cf = MLP_Node(model, layer=1, position=position, msg_cache=msg_cache, cf_cache=cf_cache, patch_type="counterfactual")

        message = torch.rand_like(msg_cache[f"blocks.1.hook_resid_mid"])

        out_clean = node.forward(message)
        out_cf = node_cf.forward(message)

        if position is None:
            expected = msg_cache[f"blocks.1.hook_mlp_out"].detach().clone() - model.blocks[1].mlp(model.blocks[1].ln1(msg_cache['blocks.1.hook_resid_mid'] - message))
        else:
            expected = torch.zeros_like(msg_cache[f"blocks.1.hook_mlp_out"])
            expected[:,position,:] = msg_cache[f"blocks.1.hook_mlp_out"][:,position,:].detach().clone() - model.blocks[1].mlp(model.blocks[1].ln1(msg_cache['blocks.1.hook_resid_mid'][:, position, :].detach().clone() - message[:, position, :]))
        
        assert torch.allclose(out_clean, expected, atol=1e-5), f"MLP_Node forward with message failed. Expected {expected}, got {out_clean}"
        assert torch.allclose(out_cf, expected, atol=1e-5), f"MLP_Node forward with message and counterfactual patch failed. Expected {expected}, got {out_cf}"
    
    elif node_class == ATTN_Node:
        null_cache = {k: torch.zeros_like(v) for k, v in msg_cache.items()}
        for head in [None, 0]:
            for keyvalue_position in [None, 1]:
                # We simply test that removing the zero message gives the same result as the clean run
                for patch_query in [True, False]:
                    for patch_key in [True, False]:
                        for patch_value in [True, False]:
                            if not (patch_query or patch_key or patch_value):
                                null_message = torch.zeros_like(msg_cache[f"blocks.1.hook_resid_pre"])
                                node = ATTN_Node(model, layer=1, head=head, position=position, msg_cache=msg_cache, cf_cache=cf_cache, keyvalue_position=keyvalue_position, patch_type="zero", patch_query=patch_query, patch_key=patch_key, patch_value=patch_value)
                                node_cf = ATTN_Node(model, layer=1, head=head, position=position, msg_cache=msg_cache, cf_cache=cf_cache, keyvalue_position=keyvalue_position, patch_type="counterfactual", patch_query=patch_query, patch_key=patch_key, patch_value=patch_value)
                                out_clean = node.forward(null_message)
                                out_cf = node_cf.forward(null_message)
                                expected = torch.zeros_like(msg_cache[f"blocks.1.hook_attn_out"])
                                assert torch.allclose(out_clean, expected, atol=1e-5), f"ATTN_Node patch_qkv{(patch_query, patch_key, patch_value)} {head} {keyvalue_position} forward with zero patch failed. Expected {expected}, got {out_clean}"
                                assert torch.allclose(out_cf, expected, atol=1e-5), f"ATTN_Node patch_qkv{(patch_query, patch_key, patch_value)} {head} {keyvalue_position} forward with counterfactual patch failed. Expected {expected}, got {out_cf}"
        # We test that removing the whole message is equal to having a zero input
        node_null = ATTN_Node(model, layer=1, head=None, position=position, msg_cache=null_cache, cf_cache=cf_cache, keyvalue_position=None, patch_type="zero", patch_query=True, patch_key=True, patch_value=True)
        node = ATTN_Node(model, layer=1, head=None, position=position, msg_cache=msg_cache, cf_cache=cf_cache, keyvalue_position=None, patch_type="zero", patch_query=True, patch_key=True, patch_value=True)
        node_cf = ATTN_Node(model, layer=1, head=None, position=position, msg_cache=msg_cache, cf_cache=cf_cache, keyvalue_position=None, patch_type="counterfactual", patch_query=True, patch_key=True, patch_value=True)
        expected = msg_cache[f"blocks.1.hook_attn_out"].detach().clone()
        if position is not None:
            expected = torch.zeros_like(msg_cache[f"blocks.1.hook_attn_out"])
            expected[:,position,:] = msg_cache[f"blocks.1.hook_attn_out"][:, position, :].detach().clone()
        expected = expected + node_null.forward(torch.zeros_like(msg_cache[f"blocks.1.hook_resid_pre"]))
        out_clean = node.forward(msg_cache[f"blocks.1.hook_resid_pre"])
        out_cf = node_cf.forward(msg_cache[f"blocks.1.hook_resid_pre"])
        assert torch.allclose(out_clean, expected, atol=1e-5), f"ATTN_Node forward with full message failed. Expected {expected}, got {out_clean}"
        assert torch.allclose(out_cf, out_clean, atol=1e-12), f"ATTN_Node forward with full message and counterfactual patch failed. Expected {expected}, got {out_cf}"

    elif node_class == EMBED_Node:
        node = EMBED_Node(model, layer=0, position=position, msg_cache=msg_cache, cf_cache=cf_cache)
        message = torch.rand_like(msg_cache["hook_embed"])
        if position is None:
            expected = message
        else:
            expected = torch.zeros_like(msg_cache["hook_embed"])
            expected[:,position,:] = message[:, position, :].detach().clone()
        assert torch.allclose(node.forward(message), expected, atol=1e-5), f"EMBED_Node forward failed. Expected {expected}, got {node.forward(message)}"

        
    
    elif node_class == FINAL_Node:
        node = FINAL_Node(model, layer=model.cfg.n_layers-1, position=position, msg_cache=msg_cache, cf_cache=cf_cache)
        message = torch.rand_like(msg_cache[f"blocks.{model.cfg.n_layers-1}.hook_resid_post"])
        if position is None:
            expected = message
        else:
            expected = torch.zeros_like(msg_cache["hook_embed"])
            expected[:,position,:] = message[:, position, :].detach().clone()
        assert torch.allclose(node.forward(message), expected, atol=1e-5), f"FINAL_Node forward failed. Expected {expected}, got {node.forward(message)}"

    else:
        raise AssertionError("Test did not run any assertions.")



@pytest.mark.parametrize("node_class", [MLP_Node, ATTN_Node, FINAL_Node, EMBED_Node])
@pytest.mark.parametrize("position", [None, 10])
@pytest.mark.parametrize("include_heads", [True, False])
@pytest.mark.parametrize("separate_kv", [True, False])
def test_get_previous_node(node_class, gpt2_model, position, include_heads, separate_kv):
    model = gpt2_model
    nodes_per_attn = model.cfg.n_heads if include_heads else 1
    kv_per_attn = 2 if separate_kv else 1

    if node_class == MLP_Node:
        layer = model.cfg.n_layers-1
        node = MLP_Node(model, layer=layer, position=position)
        predecessors = node.get_expansion_candidates(model_cfg=model.cfg, include_head=include_heads, separate_kv=separate_kv)

        expected_num_predecessors = 0
        expected_num_predecessors += 1 # EMBED_Node
        expected_num_predecessors += 1*(layer) # MLP_Nodes (only difference with FINAL_Node)
        if position is not None:
            expected_num_predecessors += (position+1)*nodes_per_attn*(layer+1)*(kv_per_attn) # ATTN_Nodes key value patched
            expected_num_predecessors += nodes_per_attn*(layer+1) # ATTN_Nodes query patched
        else:
            expected_num_predecessors += nodes_per_attn*(layer+1)*(kv_per_attn) 
            expected_num_predecessors += nodes_per_attn*(layer+1) # ATTN_Nodes query patched
        assert len(predecessors) == expected_num_predecessors, f"{node} get_previous_nodes failed. Expected {expected_num_predecessors} predecessors, got {len(predecessors)}"

    elif node_class == ATTN_Node:
        layer = model.cfg.n_layers-1
        for keyvalue_position in [None, 4]:
            for patch_key in [True, False]:
                for patch_value in [True, False]:
                    for patch_query in [True, False]:
                        node = ATTN_Node(model, layer=layer, head=0, position=position, keyvalue_position=keyvalue_position, patch_key=patch_key, patch_value=patch_value, patch_query=patch_query)

                        predecessors = node.get_expansion_candidates(model_cfg=model.cfg, include_head=include_heads, separate_kv=separate_kv)

                        keyvalue_predecessors = 0
                        query_predecessors = 0
                        if patch_key or patch_value: # Writing to the previous residual
                            if keyvalue_position is not None:
                                if patch_query and position == keyvalue_position:
                                    residuals = []
                                else:
                                    residuals = [keyvalue_position]
                            else:
                                if position is not None:
                                    if patch_query:
                                        residuals = range(position) # Avoid overlap if also query position is patched
                                    else:
                                        residuals = range(position+1)
                                else:
                                    if patch_query:
                                        residuals = [] # Only query position
                                    else:
                                        residuals = [0] # Only None position

                            for kv_pos in residuals:
                                keyvalue_predecessors += kv_per_attn * (kv_pos+1) * nodes_per_attn * (layer) # Kv patches in kv_pos
                                keyvalue_predecessors += 1*nodes_per_attn * layer # Query patches in kv_pos
                            
                            keyvalue_predecessors += len(residuals) * (layer) # MLP predecessors
                            keyvalue_predecessors += len(residuals) # EMBED predecessors
                        if patch_query:
                            pos = position if position is not None else 0
                            query_predecessors += kv_per_attn * (pos+1) * nodes_per_attn * (layer) # Kv patches writing in query position
                            query_predecessors += (layer) * nodes_per_attn # Query patches writing in query position
                            query_predecessors += layer # MLP predecessors writing in query position
                            query_predecessors += 1 # EMBED predecessors writing in query position
                        assert len(predecessors) == keyvalue_predecessors + query_predecessors, f"{node} get_previous_nodes failed. Expected {keyvalue_predecessors + query_predecessors} predecessors, got {len(predecessors)}"
    elif node_class == EMBED_Node:
        node = EMBED_Node(model, layer=0, position=position)
        predecessors = node.get_expansion_candidates(model_cfg=model.cfg, include_head=include_heads, separate_kv=separate_kv)
        assert len(predecessors) == 0, f"{node} get_previous_nodes failed. Expected 0 predecessors, got {len(predecessors)}"
    
    elif node_class == FINAL_Node:
        layer = model.cfg.n_layers-1
        node = FINAL_Node(model, layer=layer, position=position)
        predecessors = node.get_expansion_candidates(model_cfg=model.cfg, include_head=include_heads, separate_kv=separate_kv)

        expected_num_predecessors = 0
        expected_num_predecessors += 1 # EMBED_Node
        expected_num_predecessors += 1*(layer+1) # MLP_Nodes
        if position is not None:
            expected_num_predecessors += (position+1)*nodes_per_attn*(layer+1)*(kv_per_attn) # ATTN_Nodes key value patched
            expected_num_predecessors += nodes_per_attn*(layer+1) # ATTN_Nodes query patched
        else:
            expected_num_predecessors += nodes_per_attn*(layer+1)*(kv_per_attn) 
            expected_num_predecessors += nodes_per_attn*(layer+1) # ATTN_Nodes query patched
        assert len(predecessors) == expected_num_predecessors, f"{node} get_previous_nodes failed. Expected {expected_num_predecessors} predecessors, got {len(predecessors)}"

    else:
        raise AssertionError("Test did not run any assertions.")




@pytest.mark.parametrize("model_name", ["gpt2", "qwen"])
@pytest.mark.parametrize("node_class", [MLP_Node, ATTN_Node, EMBED_Node, FINAL_Node])
@pytest.mark.parametrize("position", [None, 2])
def test_gradient(node_class, model_name, gpt2_model, qwen_model, gpt2_msg_cache, qwen_msg_cache, gpt2_grad_cache, qwen_grad_cache, position):
    if model_name == "gpt2":
        model = gpt2_model
        msg_cache = gpt2_msg_cache
        grad_cache = gpt2_grad_cache
    elif model_name == "qwen":
        model = qwen_model
        msg_cache = qwen_msg_cache
        grad_cache = qwen_grad_cache
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if node_class == MLP_Node:
        node = MLP_Node(model, layer=1, position=position, msg_cache=msg_cache, cf_cache={}, patch_type="zero")

        gradient_precomputed = grad_cache['blocks.1.hook_resid_mid'].detach().clone() - grad_cache['blocks.1.hook_resid_post'].detach().clone()
        gradient_from_node = node.calculate_gradient(grad_outputs=grad_cache['blocks.1.hook_resid_post'].detach().clone())

        if position is not None:
            assert torch.allclose(gradient_from_node[:,:position,:], torch.zeros_like(gradient_from_node[:,:position,:]), atol=1e-5), f"{node} gradient for positions before {position} does not match precomputed gradient for positions before {position}."
            assert torch.allclose(gradient_from_node[:,position,:], gradient_precomputed[:,position,:], atol=1e-5), f"{node} gradient for position {position} does not match precomputed gradient for positions {position}."
            assert torch.allclose(gradient_from_node[:,position+1:,:], torch.zeros_like(gradient_from_node[:,position+1:,:]), atol=1e-5), f"{node} gradient for positions after {position} does not match precomputed gradient for positions after {position}."
        else:
            assert torch.allclose(gradient_from_node, gradient_precomputed, atol=1e-5), f"{node} gradient calculation failed. Expected {gradient_precomputed}, got {gradient_from_node}"

    elif node_class == ATTN_Node:
        node = ATTN_Node(model, layer=1, head=None, position=position, msg_cache=msg_cache, cf_cache={}, keyvalue_position=None, patch_type="zero", patch_query=True, patch_key=True, patch_value=True)

        gradient_precomputed = grad_cache['blocks.1.hook_resid_pre'].detach().clone() - grad_cache['blocks.1.hook_resid_mid'].detach().clone()
        gradient_from_node = node.calculate_gradient(grad_outputs=grad_cache['blocks.1.hook_resid_mid'].detach().clone())

        if position is None:
            assert torch.allclose(gradient_from_node, gradient_precomputed, atol=1e-5), f"{node} gradient calculation failed. Expected {gradient_precomputed}, got {gradient_from_node}"
            gradient_from_node = torch.zeros_like(gradient_from_node)
            for head in range(model.cfg.n_heads):
                node = ATTN_Node(model, layer=1, head=head, position=position, msg_cache=msg_cache, cf_cache={}, keyvalue_position=None, patch_type="zero", patch_query=True, patch_key=True, patch_value=True)
                gradient_from_node += node.calculate_gradient(grad_outputs=grad_cache['blocks.1.hook_resid_mid'].detach().clone())
            assert torch.allclose(gradient_from_node, gradient_precomputed, atol=1e-5), f"{node} gradient calculation failed. Expected {gradient_precomputed}, got {gradient_from_node}"
        else:
            assert torch.allclose(gradient_from_node[:,position+1:,:], torch.zeros_like(gradient_from_node[:,position+1:,:]), atol=1e-5), f"{node} gradient for positions after {position} does not match precomputed gradient for positions after {position}."

    
    elif node_class == EMBED_Node:
        node = EMBED_Node(model, layer=0, position=position, msg_cache=msg_cache, cf_cache={})

        gradient_precomputed = grad_cache['hook_embed'].detach().clone()
        gradient_from_node = node.calculate_gradient(grad_outputs=grad_cache['blocks.0.hook_resid_pre'].detach().clone())

        if position is not None:
            assert torch.allclose(gradient_from_node[:,:position,:], torch.zeros_like(gradient_from_node[:,:position,:]), atol=1e-5), f"{node} gradient for positions before {position} does not match precomputed gradient for positions before {position}."
            assert torch.allclose(gradient_from_node[:,position,:], gradient_precomputed[:,position,:], atol=1e-5), f"{node} gradient for position {position} does not match precomputed gradient for positions {position}."
            assert torch.allclose(gradient_from_node[:,position+1:,:], torch.zeros_like(gradient_from_node[:,position+1:,:]), atol=1e-5), f"{node} gradient for positions after {position} does not match precomputed gradient for positions after {position}."
        else:
            assert torch.allclose(gradient_from_node, gradient_precomputed, atol=1e-5), f"{node} gradient calculation failed. Expected {gradient_precomputed}, got {gradient_from_node}"

    elif node_class == FINAL_Node:
        metric = partial(simple_logit_diff, model=model, clean_residual=msg_cache[f'blocks.{model.cfg.n_layers - 1}.hook_resid_post'], target_token_id=model.to_single_token("!"))
        node = FINAL_Node(model, layer=model.cfg.n_layers-1, position=position, msg_cache=msg_cache, cf_cache={}, metric=metric)
        
        with torch.enable_grad():
            corr = msg_cache[f'blocks.{model.cfg.n_layers - 1}.hook_resid_post'].detach().clone()
            corr.requires_grad_(True)
            value = metric(corrupted_resid=corr)
            value.backward()
        gradient_precomputed = -corr.grad.detach().clone() # Negative because we want d(metric)/d(message) and not d(metric)/d(residual)
        gradient_from_node = node.calculate_gradient(grad_outputs=corr.detach().clone())

        if position is not None:
            assert torch.allclose(gradient_from_node[:,:position,:], torch.zeros_like(gradient_from_node[:,:position,:]), atol=1e-5), f"{node} gradient for positions before {position} does not match precomputed gradient for positions before {position}."
            assert torch.allclose(gradient_from_node[:,position,:], gradient_precomputed[:,position,:], atol=1e-5), f"{node} gradient for position {position} does not match precomputed gradient for positions {position}."
            assert torch.allclose(gradient_from_node[:,position+1:,:], torch.zeros_like(gradient_from_node[:,position+1:,:]), atol=1e-5), f"{node} gradient for positions after {position} does not match precomputed gradient for positions after {position}."
        else:
            assert torch.allclose(gradient_from_node, gradient_precomputed, atol=1e-5), f"{node} gradient calculation failed. Expected {gradient_precomputed}, got {gradient_from_node}"
