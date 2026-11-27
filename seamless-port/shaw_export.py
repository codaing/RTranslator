"""Export-safe Shaw relative attention for fairseq2 0.2.1.

The trained table is unchanged. Compute scores against the small table once,
then gather clipped relative-position scores for each query/key pair. This
avoids the traced square relative-key tensor and the failing dynamic Einsum.
"""
import types
import torch
from fairseq2.nn.transformer.shaw_attention import ShawRelativePositionSDPA
from fairseq2.nn.transformer.attention_mask import CustomAttentionMask


def forward(self, seqs, keys, key_padding_mask, values, *,
            attn_mask=None, needs_weights=False):
    # The Seamless streaming Conformer uses no relative value embedding.
    if self.rel_v_embed is not None:
        raise NotImplementedError("Relative value embeddings require a separate export")
    q = torch.arange(seqs.size(2), device=seqs.device) + keys.size(2) - seqs.size(2)
    k = torch.arange(keys.size(2), device=keys.device)
    indices = (k[None, :] - q[:, None]).clamp(
        -self.max_left_rel_pos, self.max_right_rel_pos) + self.max_left_rel_pos
    table = self.rel_k_embed.weight
    scores = torch.matmul(seqs, table.transpose(0, 1))
    indices = indices.unsqueeze(0).unsqueeze(0).expand(
        seqs.size(0), seqs.size(1), -1, -1)
    relative = torch.gather(scores, -1, indices) * (seqs.size(-1) ** -0.5)
    if attn_mask is not None:
        relative = relative + attn_mask.materialize()
    return self.inner_sdpa(seqs, keys, key_padding_mask, values,
                           attn_mask=CustomAttentionMask(relative),
                           needs_weights=needs_weights)


def patch_shaw_attention(model):
    count = 0
    for module in model.modules():
        if isinstance(module, ShawRelativePositionSDPA):
            module.forward = types.MethodType(forward, module)
            count += 1
    if not count:
        raise RuntimeError("Expected Shaw attention layers not found")
    print(f"Rewrote {count} Shaw attention layers for ONNX export")
