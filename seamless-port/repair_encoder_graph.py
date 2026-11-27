"""Repair only verified trace-frozen Shaw shape constants, preserving weights.

The full encoder trace produces Constant -> CumSum for positions and Constant
zeros for index broadcasting, despite the isolated Shaw trace being dynamic.
Derive these tensors from ONNX runtime Shape instead. Fail closed if a layer's
pattern changes; never accept fixed-length output as streaming-compatible.
"""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T


def rewrite(graph, expected_layers=24):
    nodes = list(graph.node)
    producers = {out: n for n in nodes for out in n.output}
    prefixes = [n.name[:-len("/MatMul")] for n in nodes
                if n.op_type == "MatMul"
                and "/speech_encoder/inner/layers." in n.name
                and n.name.endswith("/self_attn/sdpa/MatMul")]
    if len(prefixes) != expected_layers:
        raise RuntimeError(f"Expected {expected_layers} Shaw layers, found {len(prefixes)}")
    by_name = {n.name: n for n in nodes}
    # Move replacement ops to the end of the original dependency chain, then
    # topologically sort the graph. No weights are loaded or transposed here.
    replacements = {}
    for p in prefixes:
        rel_scores = by_name[p + "/MatMul"]
        content_scores = by_name[p + "/MatMul_1"]
        gather = by_name[p + "/GatherElements"]
        if gather.op_type != "GatherElements" or gather.input[0] != rel_scores.output[0]:
            raise RuntimeError(f"Unexpected Shaw gather: {p}")
        q = rel_scores.input[0]
        if content_scores.input[0] != q:
            raise RuntimeError(f"Unexpected content query: {p}")
        kt = content_scores.input[1]  # [N,H,D,K]
        broadcast = producers[gather.input[1]]
        if broadcast.op_type != "Add":
            raise RuntimeError(f"Unexpected index expansion: {p}")
        # Only replace literal int64 ones/zeros of the expected ranks and values.
        def literal(node, rank, value):
            if node.op_type != "Constant":
                return False
            a = next((a.t for a in node.attribute if a.name == "value"), None)
            if a is None or a.data_type != T.INT64 or len(a.dims) != rank:
                return False
            return bool(np.all(nh.to_array(a) == value))

        for suffix, tensor, dim in (("/CumSum", q, 2), ("/CumSum_1", kt, 3)):
            scan = by_name[p + suffix]
            source = producers[scan.input[0]]
            if source.op_type != "Constant":
                continue
            if not literal(source, 1, 1):
                raise RuntimeError(f"Unexpected positions constant: {source.name}")
            tag = source.name + "/dynamic"
            shape, index, length = [tag+s for s in ("/shape","/index","/length")]
            replacements[source.name] = [
                h.make_node("Shape", [tensor], [shape], name=tag+"/Shape"),
                h.make_node("Constant", [], [index], name=tag+"/Index",
                            value=nh.from_array(np.array([dim], dtype=np.int64))),
                h.make_node("Gather", [shape,index], [length], axis=0,name=tag+"/Gather"),
                h.make_node("ConstantOfShape", [length], list(source.output),
                            name=tag+"/Ones",value=nh.from_array(np.array([1],dtype=np.int64)))]
        zero_inputs = [producers[x] for x in broadcast.input if x in producers
                       and literal(producers[x], 4, 0)]
        if len(zero_inputs) > 1:
            raise RuntimeError(f"Ambiguous zero broadcast: {p}")
        if zero_inputs:
            source = zero_inputs[0]
            attr = next(a.t for a in source.attribute if a.name == "value")
            if attr.dims[-1] != 1:
                raise RuntimeError(f"Unexpected zero broadcast shape: {source.name}")
            tag = source.name + "/dynamic"
            shape, index, dims, one, target = [tag+s for s in
                    ("/shape","/index","/dims","/one","/target")]
            replacements[source.name] = [
                h.make_node("Shape", [rel_scores.output[0]], [shape],name=tag+"/Shape"),
                h.make_node("Constant", [], [index],name=tag+"/Indices",
                            value=nh.from_array(np.array([0,1,2],dtype=np.int64))),
                h.make_node("Gather",[shape,index],[dims],axis=0,name=tag+"/Gather"),
                h.make_node("Constant",[],[one],name=tag+"/One",
                            value=nh.from_array(np.array([1],dtype=np.int64))),
                h.make_node("Concat",[dims,one],[target],axis=0,name=tag+"/Concat"),
                h.make_node("ConstantOfShape",[target],list(source.output),
                            name=tag+"/Zeros",value=nh.from_array(np.array([0],dtype=np.int64)))]
    expanded = [part for node in nodes for part in replacements.get(node.name,[node])]
    available = {v.name for v in graph.input} | {v.name for v in graph.initializer} | {""}
    sorted_nodes = []
    pending = expanded
    while pending:
        deferred = []
        for n in pending:
            if all(i in available for i in n.input):
                sorted_nodes.append(n)
                available.update(n.output)
            else:
                deferred.append(n)
        if len(deferred) == len(pending):
            raise RuntimeError("Unresolved ONNX graph dependencies")
        pending = deferred
    del graph.node[:]
    graph.node.extend(sorted_nodes)
    # Existing intermediate shape hints may describe traced static dimensions.
    del graph.value_info[:]
    print(f"Replaced {len(replacements)} frozen tensors in {len(prefixes)} Shaw layers", flush=True)
    return len(replacements)


def repair(path):
    path = Path(path)
    model = onnx.load(str(path), load_external_data=False)
    rewrite(model.graph)
    tmp = path.with_suffix(".repair.onnx")
    onnx.save_model(model, str(tmp))
    onnx.checker.check_model(str(tmp))
    tmp.replace(path)
