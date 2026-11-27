"""Explicit inference-only attention for dynamic ONNX export.

Use tensor-derived lengths, typed mask values and no cached padding masks.
Reference parity is checked by the caller before exporting.
"""
import types
import torch
from fairseq2.nn.transformer.attention import NaiveSDPA, TorchSDPA
from fairseq2.nn.transformer.attention_mask import CausalAttentionMask, CustomAttentionMask


def axis(x, dim):
    return torch.arange(torch._shape_as_tensor(x)[dim], device=x.device)


def mask_tensor(mask, q, k):
    if isinstance(mask, CausalAttentionMask):
        rows, cols = axis(q, 2), axis(k, 2)
        allowed = cols[None, :] <= rows[:, None]
        if mask.attn_window_len is not None:
            allowed = allowed & (cols[None, :] >= rows[:, None] - mask.attn_window_len + 1)
        return torch.where(allowed, q.new_zeros(()), q.new_full((), float("-inf")))
    if isinstance(mask, CustomAttentionMask):
        return mask.mask
    raise NotImplementedError("Unrecognized attention mask: " + type(mask).__name__)


def attention(q, k, v, padding=None, mask=None, needs_weights=False):
    # The head width is a model constant, not a time dimension.
    scores = torch.matmul(q, k.transpose(-1, -2)) * (q.size(-1) ** -0.5)
    if mask is not None:
        scores = scores + mask_tensor(mask, q, k).to(dtype=scores.dtype)
    if padding is not None:
        allowed = axis(k, 2)[None, :] < padding.seq_lens[:, None]
        scores = torch.where(allowed[:, None, None, :], scores,
                             torch.full_like(scores, float("-inf")))
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(q)
    return torch.matmul(weights, v), weights if needs_weights else None


def forward(self, seqs, keys, key_padding_mask, values, *,
            attn_mask=None, needs_weights=False):
    if self.training:
        raise RuntimeError("Inference-only attention")
    return attention(seqs, keys, values, key_padding_mask, attn_mask, needs_weights)


def patch_sdpa(model):
    count = 0
    for module in model.modules():
        if isinstance(module, (TorchSDPA, NaiveSDPA)):
            module.forward = types.MethodType(forward, module)
            count += 1
    if not count:
        raise RuntimeError("No ordinary attention modules found")
    print(f"Patched {count} ordinary attention modules")
