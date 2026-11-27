"""Export-safe Shaw relative attention for fairseq2 0.2.1.

The trained table is unchanged. Compute scores against the small table once,
then gather clipped relative-position scores for each query/key pair. This
avoids the traced square relative-key tensor and the failing dynamic Einsum.
"""
import types
import torch
from fairseq2.nn.transformer.shaw_attention import ShawRelativePositionSDPA
from fairseq2.nn.transformer.attention_mask import CustomAttentionMask
from sdpa_export import mask_tensor, attention


def positions(x):
    # Keep the time axis connected to real activation data, avoiding scalar
    # shape conversions in the legacy TorchScript ONNX tracer.
    return torch.cumsum(torch.ones_like(x[0, 0, :, 0], dtype=torch.long), dim=0) - 1


def forward(self, seqs, keys, key_padding_mask, values, *,
            attn_mask=None, needs_weights=False):
    # The Seamless streaming Conformer uses no relative value embedding.
    if self.rel_v_embed is not None:
        raise NotImplementedError("Relative value embeddings require a separate export")
    q = positions(seqs)
    k = positions(keys)
    q = q + k[-1] - q[-1]
    indices = (k[None, :] - q[:, None]).clamp(
        -self.max_left_rel_pos, self.max_right_rel_pos) + self.max_left_rel_pos
    table = self.rel_k_embed.weight
    scores = torch.matmul(seqs, table.transpose(0, 1))
    # Expand from actual score tensor shapes rather than traced Python dimensions.
    indices = indices[None, None, :, :] + torch.zeros_like(scores[..., :1], dtype=torch.long)
    relative = torch.gather(scores, -1, indices) * (seqs.size(-1) ** -0.5)
    if attn_mask is not None:
        relative = relative + mask_tensor(attn_mask, seqs, keys)
    return attention(seqs, keys, values, key_padding_mask,
                     CustomAttentionMask(relative), needs_weights)


def patch_shaw_attention(model):
    count = 0
    for module in model.modules():
        if isinstance(module, ShawRelativePositionSDPA):
            module.forward = types.MethodType(forward, module)
            count += 1
    if not count:
        raise RuntimeError("Expected Shaw attention layers not found")
    print(f"Rewrote {count} Shaw attention layers for ONNX export")
