"""Generate a tiny deterministic ONNX test graph, not a speech model."""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper
out = Path("seamless/src/androidTest/assets")
out.mkdir(parents=True, exist_ok=True)
graph = h.make_graph([
    h.make_node("Cast", ["units"], ["float_units"], to=T.FLOAT),
    h.make_node("Mul", ["float_units", "scale"], ["scaled"]),
    h.make_node("Unsqueeze", ["scaled", "axes"], ["waveform"]),
], "synthetic_vocoder_interface", [h.make_tensor_value_info("units", T.INT64, [1, "n"])],
   [h.make_tensor_value_info("waveform", T.FLOAT, [1, 1, "n"])],
   [numpy_helper.from_array(np.array(0.01, dtype=np.float32), name="scale"),
    numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")])
model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)])
model.ir_version = 9
onnx.checker.check_model(model)
onnx.save(model, out / "test-vocoder.onnx")

# A 2-D MatMul graph exercises the Android XNNPACK execution provider.
xnn_graph=h.make_graph([
    h.make_node("MatMul",["x","weight"],["y"])],
    "xnnpack_matmul",[h.make_tensor_value_info("x",T.FLOAT,[1,16])],
    [h.make_tensor_value_info("y",T.FLOAT,[1,16])],
    [numpy_helper.from_array(np.eye(16,dtype=np.float32)*2,name="weight")])
xnn_model=h.make_model(xnn_graph,opset_imports=[h.make_opsetid("",17)])
xnn_model.ir_version=9
onnx.checker.check_model(xnn_model)
onnx.save(xnn_model,out/"test-xnnpack-matmul.onnx")

# Four-model deterministic integration fixture. These are tiny non-speech
# graphs: they exercise the same file names, tensor interfaces, streaming
# policy, external data and Java session chain as the real model pack.
import hashlib
import json
import tempfile
import zipfile

def init(name,value,dtype):
    return numpy_helper.from_array(np.array(value,dtype=dtype),name=name)

def vi(name,dtype,shape):
    return h.make_tensor_value_info(name,dtype,shape)

def save(directory,name,nodes,inputs,outputs,weights,external=False):
    m=h.make_model(h.make_graph(nodes,name,inputs,outputs,weights),
                   opset_imports=[h.make_opsetid("",17)],ir_version=9)
    onnx.checker.check_model(m)
    if external:
        onnx.save_model(m,str(directory/name),save_as_external_data=True,
                        all_tensors_to_one_file=True,location=name+".weights",size_threshold=0)
    else:
        onnx.save(m,str(directory/name))
    onnx.checker.check_model(str(directory/name))

with tempfile.TemporaryDirectory() as temp:
    d=Path(temp)
    # Test portable weight-only DequantizeLinear + external data on ARM64.
    save(d,"encoder.int8.onnx",[
        h.make_node("DequantizeLinear",["q","scale"],["weight"]),
        h.make_node("MatMul",["fbank","weight"],["encoded"])],
        [vi("fbank",T.FLOAT,[1,"frames",80])],
        [vi("encoded",T.FLOAT,[1,"frames",80])],
        [init("q",np.eye(80)*100,np.int8),init("scale",0.01,np.float32)],True)
    # Token prefix length 2 -> token 4; 3 -> token 5; >=4 -> EOS (2).
    # Features are dynamic in target length and decision is WRITE.
    logits=np.zeros((5,1,1,8),np.float32)
    for i,token in enumerate([2,2,4,5,2]):
        logits[i,0,0,token]=1
    save(d,"decoder.int8.onnx",[
        h.make_node("Shape",["tokens"],["shape"]),
        h.make_node("Gather",["shape","one"],["length"],axis=0),
        h.make_node("Min",["length","four"],["row"]),
        h.make_node("Gather",["logit_table","row"],["logits"],axis=0),
        h.make_node("Cast",["tokens"],["float_tokens"],to=T.FLOAT),
        h.make_node("Unsqueeze",["float_tokens","axis"],["features"]),
        h.make_node("Identity",["decision"],["p_choose"])],
        [vi("tokens",T.INT64,[1,"tokens"]),vi("encoded",T.FLOAT,[1,"frames",80])],
        [vi("logits",T.FLOAT,[1,1,8]),vi("features",T.FLOAT,[1,"tokens",1]),
         vi("p_choose",T.FLOAT,[1,1,1])],
        [init("one",1,np.int64),init("four",4,np.int64),
         init("axis",[2],np.int64),init("logit_table",logits,np.float32),
         init("decision",[[[0.9]]],np.float32)])
    # Three character positions: two accepted tokens and synthetic comma.
    # The comma has duration zero so a final flush must not duplicate output.
    scores=np.zeros((1,60,8),np.float32)
    scores[0,:30,5]=1
    scores[0,30:,6]=1
    save(d,"units.onnx",[
        h.make_node("Identity",["unit_scores"],["logits"]),
        h.make_node("Identity",["unit_durations"],["durations"])],
        [vi("features",T.FLOAT,[1,"tokens",1]),vi("chars",T.INT64,[1,"chars"]),
         vi("char_lens",T.INT64,[1,"tokens"])],
        [vi("logits",T.FLOAT,[1,60,8]),vi("durations",T.INT64,[1,3])],
        [init("unit_scores",scores,np.float32),init("unit_durations",[[30,30,0]],np.int64)])
    (d/"vocoder.onnx").write_bytes((out/"test-vocoder.onnx").read_bytes())
    meta={"prefix":[2,7],"eos":2,"pad":1,"unk":3,"comma":6,
          "table":[[[i],False,False] for i in range(8)]}
    (d/"tokenizer.json").write_text(json.dumps(meta),encoding="utf-8")
    (d/"sha256.json").write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in d.iterdir()}),encoding="utf-8")
    for corrupt in (False,True):
        with zipfile.ZipFile(out/("test-corrupt-pack.zip" if corrupt else "test-model-pack.zip"),"w") as z:
            for p in sorted(d.iterdir()):
                if corrupt and p.name=="tokenizer.json":
                    z.writestr(p.name,p.read_bytes()+b" ")
                else:z.write(p,p.name)

    (d/"decoder.int8.onnx").unlink()
    (d/"sha256.json").unlink()
    graph="decoder.incremental.fp16.onnx"
    save(d,graph,[
        h.make_node("Cast",["tokens"],["float_tokens"],to=T.FLOAT),
        h.make_node("Unsqueeze",["float_tokens","axis"],["features"]),
        h.make_node("Unsqueeze",["features","head_axis"],["current"]),
        h.make_node("Concat",["past_k_00","current"],["present_k_00"],axis=2),
        h.make_node("Concat",["past_v_00","current"],["present_v_00"],axis=2),
        h.make_node("Shape",["present_k_00"],["shape"]),
        h.make_node("Gather",["shape","two"],["length"],axis=0),
        h.make_node("Min",["length","four"],["row"]),
        h.make_node("Gather",["logit_table","row"],["logits"],axis=0),
        h.make_node("Identity",["decision"],["p_choose"])],
        [vi("tokens",T.INT64,[1,"current_tokens"]),vi("encoded",T.FLOAT,[1,"frames",80]),
         vi("position",T.INT64,[1]),vi("past_k_00",T.FLOAT,[1,1,"past",1]),vi("past_v_00",T.FLOAT,[1,1,"past",1])],
        [vi("logits",T.FLOAT,[1,1,8]),vi("features",T.FLOAT,[1,"current_tokens",1]),
         vi("p_choose",T.FLOAT,[1,1,1]),vi("present_k_00",T.FLOAT,[1,1,"total",1]),vi("present_v_00",T.FLOAT,[1,1,"total",1])],
        [init("two",2,np.int64),init("four",4,np.int64),init("axis",[2],np.int64),
         init("head_axis",[1],np.int64),init("logit_table",logits,np.float32),
         init("decision",[[[0.9]]],np.float32)])
    (d/"decoder.incremental.meta.json").write_text(json.dumps({"layers":1,"heads":1,"head_dim":1,
        "past_names":["past_k_00","past_v_00"],"present_names":["present_k_00","present_v_00"]}))
    (d/"sha256.json").write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir()}))
    with zipfile.ZipFile(out/"test-incremental-pack.zip","w") as z:
        for p in sorted(d.iterdir()):z.write(p,p.name)

    # Use actual FP16 native projection results as decoder inputs. No Java
    # conversion to float arrays; the graph consumes the tensor on every step.
    (d/"sha256.json").unlink()
    save(d,"decoder.source-projections.onnx",[
        h.make_node("Cast",["encoded"],["projected"],to=T.FLOAT16)],
        [vi("encoded",T.FLOAT,[1,"frames",80])],
        [vi("projected",T.FLOAT16,[1,"frames",80])],[])
    step=onnx.load(str(d/graph))
    step.graph.input[1].CopyFrom(vi("projected",T.FLOAT16,[1,"frames",80]))
    for node in step.graph.node:
        if node.output[0]=="logits":node.output[0]="raw_logits"
    step.graph.initializer.append(init("zero",0,np.float32))
    step.graph.node.extend([
        h.make_node("Cast",["projected"],["projected_float"],to=T.FLOAT),
        h.make_node("ReduceSum",["projected_float"],["sum"],keepdims=0),
        h.make_node("Mul",["sum","zero"],["zero_sum"]),
        h.make_node("Add",["raw_logits","zero_sum"],["logits"])])
    onnx.checker.check_model(step)
    onnx.save(step,str(d/"decoder.cached-step.onnx"))
    (d/"sha256.json").write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir()}))
    with zipfile.ZipFile(out/"test-projected-pack.zip","w") as z:
        for p in sorted(d.iterdir()):z.write(p,p.name)

# Probe the actual Android ORT contrib operator with known packed weights.
# Input K=32, N=2, block=16. Unsigned symmetric midpoint is 8 or 128.
# Output columns have weights +0.25 and -0.5. No explicit zero points.
for bits in (4,8):
    mid=1<<(bits-1)
    q=np.empty((2,2,16),np.uint8)
    q[0,:,:]=mid+1
    q[1,:,:]=mid-2
    packed=(q[:,:,0::2] | (q[:,:,1::2]<<4)) if bits==4 else q
    model=h.make_model(h.make_graph([
        h.make_node("MatMulNBits",["x","packed","scales"],["y"],domain="com.microsoft",
                    K=32,N=2,bits=bits,block_size=16,accuracy_level=0)],
        "native_lowbit_probe",[vi("x",T.FLOAT,["rows",32])],
        [vi("y",T.FLOAT,["rows",2])],
        [init("packed",packed,np.uint8),init("scales",[0.25]*4,np.float32)]),
        opset_imports=[h.make_opsetid("",17),h.make_opsetid("com.microsoft",1)],ir_version=9)
    onnx.checker.check_model(model)
    onnx.save(model,out/f"test-matmul-{bits}bit.onnx")
