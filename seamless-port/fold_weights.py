"""Expose initializer-backed RHS transforms for ORT dynamic quantization.

Only fold constant weight paths; stream newly folded tensors to an external
file, keeping the original model and its weights untouched.
"""
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper as nh, helper as h
from onnx.external_data_helper import set_external_data


def prepare(path):
    path=Path(path)
    model=onnx.load(str(path),load_external_data=False)
    graph=model.graph
    initializers={x.name:x for x in graph.initializer}
    producers={o:n for n in graph.node for o in n.output}
    supported={"Transpose","Identity","Cast"}
    def resolve(name):
        if name in initializers:
            return nh.to_array(initializers[name],base_dir=str(path.parent))
        n=producers.get(name)
        if n is None or n.op_type not in supported or len(n.input)!=1:
            return None
        value=resolve(n.input[0])
        if value is None:return None
        attrs={a.name:h.get_attribute_value(a) for a in n.attribute}
        if n.op_type=="Transpose":return value.transpose(attrs.get("perm"))
        if n.op_type=="Cast":return value.astype(h.tensor_dtype_to_np_dtype(attrs["to"]))
        return value

    targets=sorted({n.input[1] for n in graph.node if n.op_type in ("MatMul","Gemm") and len(n.input)>1 and n.input[1] not in initializers})
    folded=set()
    external=path.stem+".folded.weights"
    with (path.parent/external).open("wb") as out:
        for name in targets:
            value=resolve(name)
            if value is None or value.ndim!=2 or value.dtype!=np.float32:continue
            value=np.ascontiguousarray(value)
            offset=out.tell()
            out.write(memoryview(value).cast("B"))
            # set_external_data requires a raw_data field; do not serialize a
            # duplicate multi-gigabyte buffer into the graph.
            tensor=onnx.TensorProto(name=name,data_type=onnx.TensorProto.FLOAT,dims=value.shape,raw_data=b"")
            set_external_data(tensor,location=external,offset=offset,length=value.nbytes)
            tensor.ClearField("raw_data")
            tensor.data_location=onnx.TensorProto.EXTERNAL
            graph.initializer.append(tensor)
            folded.add(name)
    if not folded:
        (path.parent/external).unlink()
        print("No additional constant RHS transforms; using original graph",flush=True)
        return path
    # Each folded output now has an initializer. Remove its producer only;
    # other uses of the original constant are preserved.
    keep=[n for n in graph.node if not any(o in folded for o in n.output)]
    del graph.node[:];graph.node.extend(keep)
    # Prune dead transform chains and unreferenced float weights to reduce
    # quantizer memory; preserve graph outputs and subgraph dependencies.
    if not any(a.type in (onnx.AttributeProto.GRAPH,onnx.AttributeProto.GRAPHS) for n in graph.node for a in n.attribute):
        live={o.name for o in graph.output}
        keep=[]
        for n in reversed(graph.node):
            if any(o in live for o in n.output):
                keep.append(n);live.update(n.input)
        del graph.node[:];graph.node.extend(reversed(keep))
        live.update(v.name for v in graph.input)
        kept=[t for t in graph.initializer if t.name in live]
        del graph.initializer[:];graph.initializer.extend(kept)
    del graph.value_info[:]
    result=path.with_name(path.stem+".folded.onnx")
    onnx.save_model(model,str(result))
    onnx.checker.check_model(str(result))
    print(f"Exposed {len(folded)} constant matrix weights",flush=True)
    return result
