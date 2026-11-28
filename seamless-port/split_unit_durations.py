"""Extract only the ancestors of UnitY2 durations, keeping external weights.

No numerical transformation; this is a preflight experiment. The complete
unit graph still runs for eligible output. A rejected preflight cannot alter
the duration cursor. Compare end-to-end tokens, decisions and PCM before use.
"""
import copy
from pathlib import Path
import numpy as np
import onnx
from split_source_projections import deps

def should_decode(durations,offset,finished):
    d=np.asarray(durations)
    if d.ndim!=2 or d.shape[0]!=1 or not np.issubdtype(d.dtype,np.integer):
        raise ValueError("invalid duration tensor")
    if offset<0 or offset>d.shape[1] or np.any(d<0):
        raise ValueError("invalid duration context")
    return int(d[0,offset:].sum()) >= (1 if finished else 50)

def extract(model):
    g=model.graph
    if g.sparse_initializer:raise ValueError("sparse initializers unsupported")
    outputs=[v for v in g.output if v.name=="durations"]
    if len(outputs)!=1:raise ValueError("missing duration output")
    producers={o:i for i,n in enumerate(g.node) for o in n.output if o}
    keep=set();seen=set();todo=["durations"]
    while todo:
        x=todo.pop()
        if x in seen:continue
        seen.add(x)
        i=producers.get(x)
        if i is not None:
            keep.add(i);todo.extend(deps(g.node[i]))
    if not keep or len(keep)==len(g.node):
        raise ValueError("duration graph has no independent acoustic tail")
    nodes=[g.node[i] for i in sorted(keep)]
    used=set().union(*(deps(n) for n in nodes))
    defined={v.name for v in g.input}|{v.name for v in g.initializer}
    defined.update(o for n in nodes for o in n.output)
    if not used.issubset(defined):raise ValueError("unresolved duration dependency")
    result=copy.deepcopy(model)
    result.graph.CopyFrom(onnx.helper.make_graph(nodes,"duration_only",
        list(g.input),outputs,[t for t in g.initializer if t.name in used or t.name in {v.name for v in g.input}]))
    return result,{"full_nodes":len(g.node),"duration_nodes":len(keep),
                   "removed_nodes":len(g.node)-len(keep)}

def main(directory):
    import json
    d=Path(directory)
    m=onnx.load(str(d/"units.onnx"),load_external_data=False)
    result,report=extract(m)
    target=d/"units.duration-only.onnx"
    onnx.save_model(result,str(target))
    onnx.checker.check_model(str(target))
    print(json.dumps(report),flush=True)

if __name__=="__main__":
    import sys
    main(sys.argv[1])
