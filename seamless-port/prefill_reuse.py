"""Approximate across-source-update decoder reuse. Never use by default."""
import numpy as np

def retained_prefix(tokens,previous,tail,finished):
    if not tail or finished or previous is None:return 0
    if tuple(tokens)!=tuple(previous):
        raise ValueError("saved decoder prefix differs from accepted tokens")
    return max(0,len(tokens)-tail)

def retain_state(caches,features,cut):
    if not caches or features is None or cut<1 or cut>len(features):
        raise ValueError("missing saved prefix")
    length=len(features)
    if any(c.ndim!=4 or c.shape[0]!=1 or c.shape[2]!=length for c in caches):
        raise ValueError("saved cache does not align with accepted features")
    return [np.ascontiguousarray(c[:,:,:cut,:]) for c in caches],list(features[:cut])
