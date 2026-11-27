"""On a failed encoder export, record first attention graph dependencies and inferred shapes.

No extra full model session or external weight loading. Reports are text only.
"""
import traceback
import onnx


def inspect_failure(path, feeds, directory):
    report = directory / "encoder-graph-diagnostics.txt"
    try:
        model = onnx.load(str(path), load_external_data=False)
        nodes = list(model.graph.node)
        producers = {o:n for n in nodes for o in n.output}
        types = {v.name: str(v.type.tensor_type.shape).replace("\n", " ")
                 for v in list(model.graph.input)+list(model.graph.value_info)+list(model.graph.output)}
        constants = {v.name: list(v.dims) for v in model.graph.initializer}
        selected = [n for n in nodes if "/speech_encoder/inner/layers.0/self_attn/sdpa/" in n.name]
        visited = set()
        with report.open("w") as out:
            print("Failed input:", {k:list(v.shape) for k,v in feeds.items()}, file=out)
            def visit(n, depth):
                if n.name in visited:
                    return
                visited.add(n.name)
                print(n.name, n.op_type, file=out)
                for inp in n.input:
                    print("  input:",inp,"shape:",types.get(inp,constants.get(inp,"unknown")),file=out)
                    if depth and inp in producers:
                        visit(producers[inp], depth-1)
                for attr in n.attribute:
                    if attr.name == "value" and attr.HasField("t"):
                        t = attr.t
                        if t.data_location == onnx.TensorProto.DEFAULT and len(t.raw_data) < 128 and all(d < 32 for d in t.dims):
                            try:
                                print("  constant:",onnx.numpy_helper.to_array(t).tolist(),file=out)
                            except Exception:
                                pass
                print("  outputs:",list(n.output),file=out)
            for n in selected:
                visit(n,5)
        print("Wrote",report,flush=True)
    except Exception:
        report.write_text(traceback.format_exc())
