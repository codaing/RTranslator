"""Hoist encoder-only decoder computations into a per-source-update graph.

Both graphs reference the same original external weights, without loading them
into Python or rewriting precision. No target-conditioned node can move.
Experimental: strict graph/type checks and real speech comparison required.
"""
import copy
import json
from pathlib import Path
import onnx
from onnx import helper as h

def free_inputs(graph):
    defined={x.name for x in graph.input}|{x.name for x in graph.initializer}
    defined.update(o for n in graph.node for o in n.output if o)
    used=set()
    for n in graph.node:used.update(deps(n))
    used.update(o.name for o in graph.output)
    return used-defined

def deps(node):
    names={x for x in node.input if x}
    for a in node.attribute:
        if a.type==onnx.AttributeProto.GRAPH:names.update(free_inputs(a.g))
        elif a.type==onnx.AttributeProto.GRAPHS:
            for g in a.graphs:names.update(free_inputs(g))
    return names

def infer_types(model,base_dir):
    # The real FP16 export has externalized small integer reshape operands.
    # Load only small shape tensors into an isolated metadata copy, never
    # the large floating-point checkpoint weights.
    from onnx.external_data_helper import load_external_data_for_tensor
    from model_pack import tensors
    from onnx import TensorProto as T
    import math
    work=copy.deepcopy(model)
    count=0
    for t in tensors(work.graph):
        if t.data_location==T.EXTERNAL and t.data_type in (T.INT64,T.INT32,T.BOOL) and math.prod(t.dims)<=65536:
            load_external_data_for_tensor(t,str(base_dir))
            t.ClearField("external_data");t.data_location=T.DEFAULT
            count+=1
    print("loaded external shape tensors",count,flush=True)
    inferred=onnx.shape_inference.infer_shapes(work,strict_mode=False,data_prop=True)
    types={v.name:v for v in list(inferred.graph.input)+list(inferred.graph.value_info)+list(inferred.graph.output)}
    return work,types

def explicit_transpose_type(graph,name,types,producers):
    """Recover a transpose's exact rank, without guessing dimension sizes.

    ONNX Transpose with an explicit perm has rank len(perm), independent
    of upstream dynamic Reshape values. Trace only ops whose output dtype
    is specified by their first input (or Cast's explicit target type).
    """
    g=graph
    i=producers.get(name)
    if i is None:return None
    node=g.node[i]
    if node.domain not in ("","ai.onnx") or node.op_type!="Transpose":return None
    perm=next((list(a.ints) for a in node.attribute if a.name=="perm"),None)
    if perm is None or sorted(perm)!=list(range(len(perm))):return None
    initializers={t.name:t.data_type for t in g.initializer}
    inputs={v.name:v.type.tensor_type.elem_type for v in g.input}
    preserving={"Transpose","Reshape","MatMul","Gemm","Add","Sub","Mul","Div",
                "Identity","Relu","Tanh","Sigmoid","Squeeze","Unsqueeze",
                "Gather","Slice","Expand","Flatten","Concat","Neg","Pow",
                "Sqrt","Erf","Clip","Softmax"}
    def dtype(tensor,seen):
        if tensor in seen:return 0
        vi=types.get(tensor)
        if vi is not None and vi.type.tensor_type.elem_type:
            return vi.type.tensor_type.elem_type
        if tensor in initializers:return initializers[tensor]
        if tensor in inputs:return inputs[tensor]
        j=producers.get(tensor)
        if j is None:return 0
        n=g.node[j]
        if n.domain not in ("","ai.onnx"):return 0
        if n.op_type=="Cast":
            return next((a.i for a in n.attribute if a.name=="to"),0)
        if n.op_type=="Constant":
            return next((a.t.data_type for a in n.attribute if a.name=="value"
                         and a.type==onnx.AttributeProto.TENSOR),0)
        if n.op_type in preserving and n.input:
            return dtype(n.input[0],seen|{tensor})
        return 0
    elem=dtype(name,set())
    if not elem:return None
    # Do not infer fixed dimensions from a sample or invent symbolic sizes.
    return h.make_tensor_value_info(name,elem,[None]*len(perm))

def partition(model,base_dir="."):
    g=model.graph
    initializers={x.name:x for x in g.initializer}
    inputs={x.name:x for x in g.input}
    if "encoded" not in inputs:raise ValueError("encoded input missing")
    # Taint tracking: constant=0, source=1, target=2, mixed=3.
    taint={name:0 for name in initializers}
    taint.update({name:1 if name=="encoded" else 2 for name in inputs})
    source_nodes=set()
    for i,n in enumerate(g.node):
        names=deps(n)
        if not names.issubset(taint):
            raise ValueError(f"unresolved inputs for {n.name}: {names-set(taint)}")
        mask=0
        for name in names:mask|=taint[name]
        # Do not hoist nodes with control-flow graphs or nondeterministic ops.
        if any(a.type in (onnx.AttributeProto.GRAPH,onnx.AttributeProto.GRAPHS) for a in n.attribute) or n.op_type.startswith("Random") or n.op_type in ("Dropout","Multinomial"):
            mask|=2
        if mask==1:source_nodes.add(i)
        for name in n.output:
            if name:taint[name]=mask
    if not source_nodes:raise ValueError("no source-only computations")
    # Pull constant-only producers into the source graph when needed, but keep
    # any shared producer in the main graph as well.
    producers={o:i for i,n in enumerate(g.node) for o in n.output if o}
    main_nodes=set(range(len(g.node)))-source_nodes
    source_needed=set(source_nodes)
    pending=list(source_nodes)
    while pending:
        i=pending.pop()
        for x in deps(g.node[i]):
            j=producers.get(x)
            if j is not None and j not in source_needed:
                if taint[x]!=0:raise ValueError("nonconstant dependency in hoisted graph")
                source_needed.add(j);pending.append(j)
    main_used=set().union(*(deps(g.node[i]) for i in main_nodes)) if main_nodes else set()
    main_used.update(x.name for x in g.output)
    source_outputs={o for i in source_nodes for o in g.node[i].output if o}
    boundary=sorted(source_outputs&main_used)
    if not boundary:raise ValueError("no source projection boundary")
    # Remove constant-only dead branches from main, but preserve shared ones.
    required=set()
    pending=list(main_used & {x.name for x in g.output})
    while pending:
        x=pending.pop();i=producers.get(x)
        if i is None or i in source_nodes or i in required:continue
        required.add(i);pending.extend(deps(g.node[i]))
    main_nodes=required
    boundary=sorted(source_outputs & (set().union(*(deps(g.node[i]) for i in main_nodes))|{x.name for x in g.output}))
    if not boundary:raise ValueError("no live source projection boundary")
    work,types=infer_types(model,base_dir)
    def valid(v):
        return v is not None and v.type.tensor_type.elem_type and v.type.tensor_type.HasField("shape")
    missing=[name for name in boundary if not valid(types.get(name))]
    recovered=[]
    for name in missing:
        vi=explicit_transpose_type(g,name,types,producers)
        if vi is not None:
            types[name]=vi
            recovered.append(name)
    if recovered:
        print("recovered explicit transpose ranks",len(recovered),flush=True)
    missing=[name for name in boundary if not valid(types.get(name))]
    if missing:
        print("standard and transpose inference missing",missing,flush=True)
        # Isolate the source subgraph, with only the original encoded input.
        # Symbolic inference can attempt to evaluate floating-point weights
        # (e.g. an Add bias), even when only shapes are needed. Represent all
        # remaining external initializers as typed inputs in this metadata-only
        # probe. Small integer shape operands were materialized above.
        # The real source and decoder graphs retain their original weights.
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
        nodes=[work.graph.node[i] for i in sorted(source_needed)]
        used=set().union(*(deps(n) for n in nodes))
        small=copy.deepcopy(work)
        probe_inputs=[inputs["encoded"]]
        probe_constants=[]
        for t in work.graph.initializer:
            if t.name not in used:continue
            if t.data_location==onnx.TensorProto.EXTERNAL:
                probe_inputs.append(h.make_tensor_value_info(t.name,t.data_type,list(t.dims)))
            else:
                probe_constants.append(t)
        small.graph.CopyFrom(h.make_graph(nodes,"source_shape_probe",probe_inputs,
            [h.make_empty_tensor_value_info(n) for n in boundary],probe_constants))
        sym=SymbolicShapeInference.infer_shapes(small,auto_merge=True,guess_output_rank=False)
        for v in list(sym.graph.value_info)+list(sym.graph.output):
            if valid(v) and not valid(types.get(v.name)):
                shape=[d.dim_value if d.HasField("dim_value") else None for d in v.type.tensor_type.shape.dim]
                types[v.name]=h.make_tensor_value_info(v.name,v.type.tensor_type.elem_type,shape)
    missing=[name for name in boundary if not valid(types.get(name))]
    if missing:raise ValueError("missing inferred boundary types: "+str(missing))
    print("boundary types",[(n,types[n].type.tensor_type.elem_type,len(types[n].type.tensor_type.shape.dim)) for n in boundary],flush=True)
    def build(indices,graph_inputs,graph_outputs,name):
        nodes=[g.node[i] for i in sorted(indices)]
        used=set().union(*(deps(n) for n in nodes))
        used.update(v.name for v in graph_outputs)
        weights=[v for v in g.initializer if v.name in used]
        new=copy.deepcopy(model)
        new.graph.CopyFrom(h.make_graph(nodes,name,graph_inputs,graph_outputs,weights))
        return new
    source=build(source_needed,[inputs["encoded"]],[types[n] for n in boundary],"source_projections")
    main_deps=set().union(*(deps(g.node[i]) for i in main_nodes))|{x.name for x in g.output}
    main_inputs=[v for v in g.input if v.name in main_deps]+[types[n] for n in boundary]
    main=build(main_nodes,main_inputs,list(g.output),"cached_decoder")
    # Require actual expensive projections, not just scalar/shape operations.
    matmuls=sum(g.node[i].op_type in ("MatMul","Gemm") for i in source_nodes)
    if matmuls<1:raise ValueError("no source-only matrix multiplications hoisted")
    report={"source_nodes":len(source_nodes),"source_matmuls":matmuls,
            "main_nodes":len(main_nodes),"boundary":boundary,
            "explicit_transpose_ranks":recovered,
            "main_inputs":[v.name for v in main_inputs]}
    return source,main,report

def main(directory):
    directory=Path(directory)
    graph=directory/"decoder.incremental.fp16.onnx"
    model=onnx.load(str(graph),load_external_data=False)
    source,decoder,report=partition(model,directory)
    for name,m in (("decoder.source-projections.onnx",source),("decoder.cached-step.onnx",decoder)):
        path=directory/name
        onnx.save_model(m,str(path))
        onnx.checker.check_model(str(path))
    (directory/"decoder.source-projections.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)

if __name__=="__main__":
    import sys
    main(sys.argv[1])
