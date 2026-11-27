"""Build a storage/CPU-compatibility FP16 candidate for FP32 acoustic graphs."""
import json
from pathlib import Path
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common.float16 import convert_float_to_float16, DEFAULT_OP_BLOCK_LIST


def nrms(actual,expected):
    a=np.asarray(actual,dtype=np.float64);e=np.asarray(expected,dtype=np.float64)
    return float(np.sqrt(np.mean((a-e)**2))/max(float(np.sqrt(np.mean(e**2))),1e-6))




def fix_mixed_mul_types(graph,inherited=None):
    """Infer internal floating types and cast FP32 operands of mixed FP16 Mul nodes."""
    dtypes=dict(inherited or {})
    shapes={}
    for vi in list(graph.input)+list(graph.value_info)+list(graph.output):
        tt=vi.type.tensor_type
        if tt.elem_type:
            dtypes[vi.name]=tt.elem_type
            shapes[vi.name]=[d.dim_value if d.HasField("dim_value") else None for d in tt.shape.dim] if tt.HasField("shape") else None
    for tensor in graph.initializer:
        dtypes[tensor.name]=tensor.data_type
        shapes[tensor.name]=list(tensor.dims)
    patched=0;new_nodes=[]
    binary_float_ops={"Add","Sub","Mul","Div","Pow","MatMul","Gemm"}
    for node in graph.node:
        # Infer nested subgraph input types from values already seen in the parent.
        for attr in node.attribute:
            if attr.type==onnx.AttributeProto.GRAPH:
                patched+=fix_mixed_mul_types(attr.g,dtypes)
            elif attr.type==onnx.AttributeProto.GRAPHS:
                for subgraph in attr.graphs:patched+=fix_mixed_mul_types(subgraph,dtypes)
        if node.op_type=="Constant":
            for attr in node.attribute:
                if attr.name=="value" and attr.type==onnx.AttributeProto.TENSOR:
                    dtypes[node.output[0]]=attr.t.data_type
                    shapes[node.output[0]]=list(attr.t.dims)
        casts=[]
        if node.op_type in binary_float_ops:
            known=[dtypes.get(name) for name in node.input]
            if onnx.TensorProto.FLOAT in known and onnx.TensorProto.FLOAT16 in known:
                for index,dtype in enumerate(known):
                    if dtype==onnx.TensorProto.FLOAT:
                        original=node.input[index]
                        cast_out=f"{original}__to_fp16_for_{node.name.replace('/','_')}_{index}"
                        cast_name=f"cast_fp16_for_{node.name.replace('/','_')}_{index}"
                        casts.append(onnx.helper.make_node("Cast",[original],[cast_out],
                            to=onnx.TensorProto.FLOAT16,name=cast_name))
                        dtypes[cast_out]=onnx.TensorProto.FLOAT16
                        shapes[cast_out]=shapes.get(original)
                        graph.value_info.append(onnx.helper.make_tensor_value_info(
                            cast_out,onnx.TensorProto.FLOAT16,shapes.get(original)))
                        node.input[index]=cast_out
                        patched+=1
                if node.output:
                    dtypes[node.output[0]]=onnx.TensorProto.FLOAT16
                    for vi in graph.value_info:
                        if vi.name==node.output[0]:
                            vi.type.tensor_type.elem_type=onnx.TensorProto.FLOAT16
        new_nodes.extend(casts)
        new_nodes.append(node)
        if node.op_type=="Cast":
            to=next((a.i for a in node.attribute if a.name=="to"),None)
            if to is not None and node.output:dtypes[node.output[0]]=to
        elif node.op_type=="Constant":
            pass
        else:
            # Most neural-network operators preserve the floating element type
            # of their data input. This propagates types through MatMul/Transpose
            # and lets us detect mixed-type scalar operands downstream.
            floating=[dtypes.get(name) for name in node.input
                      if dtypes.get(name) in (onnx.TensorProto.FLOAT,onnx.TensorProto.FLOAT16)]
            if floating:
                inferred=floating[0]
                for out in node.output:
                    if out not in dtypes:dtypes[out]=inferred
    graph.ClearField("node")
    graph.node.extend(new_nodes)
    return patched

def main(component,directory):
    if component not in ("units","vocoder"):raise ValueError("FP16 experiment is only for units/vocoder")
    directory=Path(directory);source=directory/f"{component}.onnx"
    output=directory/f"{component}.fp16.onnx";marker=directory/f"{component}.fp16.validated.json"
    marker.unlink(missing_ok=True)
    result={"component":component,"precision":"fp16-candidate","parity":"PENDING","tests":[]}
    try:
        if not (directory/f"{component}.validated.json").exists():raise RuntimeError("validated FP32 export is required")
        fixtures=sorted(directory.glob(f"{component}.case*.npz"))
        if not fixtures:raise RuntimeError("independent FP32 fixtures are missing")
        base_graph=onnx.load(str(source),load_external_data=False)
        base_locations=set()
        for t in base_graph.graph.initializer:
            for e in t.external_data:
                if e.key=="location":base_locations.add(e.value)
        result["fp32_bytes"]=source.stat().st_size+sum((directory/x).stat().st_size for x in base_locations)
        del base_graph
        model=onnx.load(str(source),load_external_data=True)
        # Preserve fused scaled-dot-product attention subgraphs in FP32.
        # SDPA nodes can be nested inside graph-valued attributes, so walk the
        # whole graph tree rather than only the top-level node list.
        def walk(graph):
            for node in graph.node:
                yield node
                for attr in node.attribute:
                    if attr.type==onnx.AttributeProto.GRAPH:
                        yield from walk(attr.g)
                    elif attr.type==onnx.AttributeProto.GRAPHS:
                        for subgraph in attr.graphs:yield from walk(subgraph)
        blocked=[n.name for n in walk(model.graph) if "/sdpa/" in n.name.lower()]
        result["blocked_sdpa_nodes"]=len(blocked)
        start=time.monotonic()
        model=convert_float_to_float16(model,keep_io_types=True,disable_shape_infer=True,
                                       op_block_list=list(DEFAULT_OP_BLOCK_LIST)+["Mul"],
                                       node_block_list=blocked)
        result["conversion_seconds"]=time.monotonic()-start
        result["mixed_mul_casts"]=fix_mixed_mul_types(model.graph)
        weights=directory/(output.name+".weights")
        weights.unlink(missing_ok=True)
        onnx.save_model(model,str(output),save_as_external_data=True,all_tensors_to_one_file=True,
            location=output.name+".weights",size_threshold=1024)
        onnx.checker.check_model(str(output))
        result["fp16_bytes"]=output.stat().st_size+(weights.stat().st_size if weights.exists() else 0)
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(output),sess_options=options,providers=["CPUExecutionProvider"])
        failures=[]
        for path in fixtures:
            with np.load(path,allow_pickle=False) as f:
                inputs={i.name:f["input_"+i.name] for i in session.get_inputs()}
                names=[o.name for o in session.get_outputs()]
                values=session.run(names,inputs);metrics={"fixture":path.name}
                for name,actual in zip(names,values):
                    expected=f["output_"+name]
                    error=nrms(actual,expected);metrics[name+"_nrms"]=error
                    if actual.shape!=expected.shape or not np.isfinite(actual).all():
                        failures.append(path.name+": invalid "+name+" output")
                    elif component=="units" and name=="durations":
                        if not np.array_equal(actual,expected):failures.append(path.name+": duration mismatch")
                    elif component=="units" and name=="logits":
                        agreement=float(np.mean(actual.argmax(-1)==expected.argmax(-1)))
                        metrics["unit_argmax_agreement"]=agreement
                        if error>0.10 or agreement!=1.0:failures.append(path.name+": unit logits/argmax mismatch")
                    elif component=="vocoder":
                        metrics[name+"_max_abs"]=float(np.max(np.abs(actual-expected)))
                        if error>0.02 or metrics[name+"_max_abs"]>0.05:failures.append(path.name+": waveform tolerance exceeded")
                result["tests"].append(metrics);print(json.dumps(metrics),flush=True)
        result["failures"]=failures
        result["parity"]="FAIL" if failures else "PASS"
        result["size_reduction_percent"]=100.0*(1-result["fp16_bytes"]/max(result["fp32_bytes"],1))
    except Exception as exc:
        result["parity"]="FAIL";result["reason"]=str(exc)
    marker.write_text(json.dumps(result,indent=2)+"\n")
    print(marker);print(json.dumps(result,indent=2),flush=True)
    if result["parity"]!="PASS":raise RuntimeError(result.get("reason") or "; ".join(result.get("failures",[])))

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("component",choices=("units","vocoder"));p.add_argument("directory",type=Path)
    a=p.parse_args();main(a.component,a.directory)
