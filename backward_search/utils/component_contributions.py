import torch
import einops
from transformer_lens import HookedTransformer, ActivationCache
from torch import Tensor
from backward_search.utils.hooks import zero_patch_except_head_hook, zero_patch_except_pos_hook, zero_patch_except_pos_batched_hook, zero_patch_except_head_batched_hook, zero_patch_except_pos_and_head_batched_hook, stack_hook, attention_score_patch_hook
from functools import partial
import traceback


def contribution_mlp(model: HookedTransformer, block: int, residual: Tensor, position=None, patch_to_remove=None) -> Tensor:
    """
    contribution_mlp calculate the contribution of the MLP to the residual stream.

    Calculates the contribution of the MLP to the residual stream by applying the MLP to the residual and returning the output.
    It performs a forward pass through the MLP at the specified layer, applying layer normalization and the MLP transformation.
    
     Args:
        model (HookedTransformer): The transformer model to which the MLP belongs.
        block (int): The block number (layer) of the MLP.
        residual (Tensor): The residual entering into the MLP.
        position (int, optional): If set the returned value will be zeroed out for all positions except this one. Defaults to None.
        patch_to_remove (Tensor, optional): A patch to remove from the residual before the forward pass. Defaults to None.

    Returns:
        Tensor: The output of the MLP after applying layer normalization and a forward pass to the residual.
    """
    layer = model.blocks[block]
    if patch_to_remove is not None:
        residual = residual - patch_to_remove
    residual = layer.ln2(residual)
    out = layer.mlp(residual)
    if position is not None:
        if position == -1:
            position = residual.shape[1] - 1
        out[:, :position, :] = torch.zeros_like(out[:, :position, :])
        out[:, position + 1:, :] = torch.zeros_like(out[:, position + 1:, :])
    return out


def contribution_attn(model: HookedTransformer, block: int, residual: Tensor, in_position=None, out_position=None, head=None, patch_to_remove=None, patch_query=False, patch_key=False, patch_value=True) -> Tensor:
    """
    contribution_attn calculate the contribution of the attention mechanism to the residual stream.

    Calculates the contribution of the attention mechanism to the residual stream by applying the attention mechanism to the residual and returning the output.
    It always performs a forward pass through the attention bloack at the specified layer, applying layer normalization and the attention transformation.
    
    Args:
        model (HookedTransformer): The transformer model to which the attention mechanism belongs.
        block (int): The block number (layer) of the attention mechanism.
        residual (Tensor): The residual entering into the attention mechanism.
        in_position (int, optional): If set only the contribution of this position will be evaluated. Defaults to None.
        out_position (int, optional): If set only the contribution towards this position will be evaluated. Defaults to None.
        head (int, optional): If set only the contribution of this head will be evaluated. Defaults to None.
        patch_to_remove (Tensor, optional): A patch to remove from the residual before the forward pass. Defaults to None.

    Returns:
        Tensor: The output of the attention block applied to the residual.
    """
    model.reset_hooks()
    layer = model.blocks[block]

    if patch_to_remove is not None:
        residual = residual - patch_to_remove

    if in_position is not None:
        if in_position == -1:
            in_position = residual.shape[1] - 1
        hook_fn = partial(zero_patch_except_pos_hook, pos=in_position)
        if patch_query:
            model.add_hook(f'blocks.{block}.attn.hook_q', hook_fn)
        if patch_key:
            model.add_hook(f'blocks.{block}.attn.hook_k', hook_fn)
        if patch_value:
            model.add_hook(f'blocks.{block}.attn.hook_v', hook_fn)

    if out_position is not None:
        if out_position == -1:
            out_position = residual.shape[1] - 1
        hook_fn = partial(zero_patch_except_pos_hook, pos=out_position)
        model.add_hook(f'blocks.{block}.attn.hook_q', hook_fn)
    
    if head is not None:
        hook_fn = partial(zero_patch_except_head_hook, head=head)
        model.add_hook(f'blocks.{block}.attn.hook_v', hook_fn)
        model.add_hook(f'blocks.{block}.attn.hook_k', hook_fn)

    residual = layer.ln1(residual)
    out = layer.attn(
        query_input=residual,
        key_input=residual,
        value_input=residual
    )
    if head is not None or in_position is not None:
        model.reset_hooks()
        out = out - model.b_O[block]

    return out

def contribution_attn_fast(model: HookedTransformer, block: int, residual: Tensor, separate_position_in=False, separate_heads=False, patch_to_remove=None, patch_query=False, patch_key=False, patch_value=True) -> dict:
    model.reset_hooks()
    layer = model.blocks[block]
    batch_size = residual.shape[0]
    seq_length = residual.shape[1]
    # if block >=9:
    # 	patch_query, patch_key, patch_value = True, False, False
    if patch_to_remove is not None:
        residual = residual - patch_to_remove
        
    if separate_position_in:
        positions = list(range(residual.shape[1]))
        if patch_query:
            model.add_hook(f'blocks.{block}.attn.hook_q', zero_patch_except_pos_batched_hook)
        else:
            model.add_hook(f'blocks.{block}.attn.hook_q', partial(stack_hook, n=seq_length))
        if patch_key:
            model.add_hook(f'blocks.{block}.attn.hook_k', zero_patch_except_pos_batched_hook)
        else:
            model.add_hook(f'blocks.{block}.attn.hook_k', partial(stack_hook, n=seq_length))
        if patch_value:
            model.add_hook(f'blocks.{block}.attn.hook_v', zero_patch_except_pos_batched_hook)
        else:
            model.add_hook(f'blocks.{block}.attn.hook_v', partial(stack_hook, n=seq_length))
    else:
        positions = ['ALL']
    
    if separate_heads:
        heads = list(range(model.cfg.n_heads))
        model.add_hook(f'blocks.{block}.attn.hook_v', zero_patch_except_head_batched_hook)
        model.add_hook(f'blocks.{block}.attn.hook_k', zero_patch_except_head_batched_hook)
        model.add_hook(f'blocks.{block}.attn.hook_q', partial(stack_hook, n=model.cfg.n_heads))
    else:
        heads = ['ALL']
        
    residual = layer.ln1(residual)
    attentions = layer.attn(
        query_input=residual,
        key_input=residual,
        value_input=residual
    )

    if separate_position_in or separate_heads:
        attentions = attentions - model.b_O[block]
        model.reset_hooks()
  
    out = {}
    for head in heads:
        if head == 'ALL':
            head_base = 0
        else:
            head_base = head * batch_size * len(positions)
        for position in positions:
            tag = f"head_{head}_pos_{position}"
            if position == 'ALL':
                pos_base = 0
            else:
                pos_base = position * batch_size
            
            out[tag] = attentions[head_base + pos_base:head_base + pos_base + batch_size]		
    return out


def contribution_attn2(model: HookedTransformer, block: int, cache: ActivationCache, in_position: int, out_position: int, head=None, patch_query=False, patch_key=False, patch_value=False, patch_to_remove=None) -> Tensor:
    """
    contribution_attn2 contribute the contribution of the attention mechanism to the residual stream when patching different parts of the attention mechanism.

    This function is used to calculate the contribution of the attention mechanism to the residual stream by applying the attention mechanism when the input is patched either in the query, key or value.
    Furthermore, it allows to patch only specific positions and it allows to separate the contribution of single attention heads.

    Args:
        model (HookedTransformer): The transformer model to which the attention belongs.
        block (int): The block number (layer) of the attention.
        cache (ActivationCache): The cache of activations saved during the forward pass.
        in_position (int): The position in the input sequence to patch (key and value).
        out_position (int): The position in the output sequence to patch (query).
        head (int, optional): The head to whose contribution we want to isolate. Defaults to None.
        patch_query (bool, optional): True if we want to patch the query input. Defaults to False.
        patch_key (bool, optional): True if we want to patch the key input. Defaults to False.
        patch_value (bool, optional): True if we want to patch the value input. Defaults to False.
        patch_to_remove (Tensor, optional): The message we want to remove from the input. Assumed to be of shape (batch_size, seq_length, d_model) Defaults to None.

    Raises:
        e: Exception: If an error occurs during the forward pass.

    Returns:
        Tensor: The output of the attention block applied to the residual. Assumed to be of shape (batch_size, seq_length, d_model).
    """
    try:
        model.reset_hooks() # reset state of the model
        residual = cache[f'blocks.{block}.hook_resid_pre'].clone()
        output = cache[f'blocks.{block}.hook_attn_out'].clone()
        if patch_to_remove is None and head is None: # If we are not patching anything we can just return the output
            return output
        layer = model.blocks[block]

        residual_key = residual.clone() # clone to avoid modifying the original residual
        residual_value = residual.clone()
        residual_query = residual.clone()

        if patch_key:
            if in_position is not None: # if we patch the key in a specific position we just modify the input residual stream
                residual_key[:, in_position, :] = residual_key[:, in_position, :] - patch_to_remove[:, in_position, :]
            else:
                residual_key = residual_key - patch_to_remove[:, :residual_key.shape[1], :]
        if patch_value:
            if in_position is not None: # if we patch the value in a specific position we just modify the input residual stream
                residual_value[:, in_position, :] = residual_value[:, in_position, :] - patch_to_remove[:, in_position, :]
            else:
                residual_value = residual_value - patch_to_remove[:, :residual_value.shape[1], :]
        if patch_query:
            if out_position is not None: # if we patch the query in a specific position we just modify the input residual stream
                residual_query[:, out_position, :] = residual_query[:, out_position, :] - patch_to_remove[:, out_position, :]
            else:
                residual_query = residual_query - patch_to_remove
        if head is not None:
            hook_fn = partial(zero_patch_except_head_hook, head=head)
            model.add_hook(f'blocks.{block}.attn.hook_v', hook_fn)
            model.add_hook(f'blocks.{block}.attn.hook_k', hook_fn)
        residual_key = layer.ln1(residual_key)
        residual_value = layer.ln1(residual_value)
        residual_query = layer.ln1(residual_query)

        out = layer.attn(
            query_input=residual_query,
            key_input=residual_key,
            value_input=residual_value
        )

        model.reset_hooks()
        return out
    except Exception as e:
        print(traceback.format_exc())
        print(f"Error in contribution_attn2: {traceback.extract_tb(e.__traceback__)}")
        raise e 

