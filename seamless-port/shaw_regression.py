"""Small fairseq2 Shaw module export; validates dynamic query/key lengths."""
import tempfile
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from torch import nn
from fairseq2.nn.transformer.shaw_attention import ShawRelativePositionSDPA
from shaw_export import patch_shaw_attention


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.sdpa = ShawRelativePositionSDPA(64, 4, 16, max_right_rel_pos=4)

    def forward(self, q, k, v):
        return self.sdpa(q, k, None, v)[0]


def main():
    torch.manual_seed(0)
    model = Model().eval()
    fixtures = [(torch.randn(1,4,q,16), torch.randn(1,4,k,16), torch.randn(1,4,k,16))
                for q,k in ((96,96),(128,128),(97,97),(3,12),(1,5))]
    with torch.inference_mode(), tempfile.TemporaryDirectory() as tmp:
        expected = [model(*f).numpy() for f in fixtures]
        patch_shaw_attention(model)
        for f,e in zip(fixtures, expected):
            np.testing.assert_allclose(model(*f).numpy(),e,rtol=0.002,atol=0.0002)
        path = str(Path(tmp)/"shaw.onnx")
        torch.onnx.export(model, fixtures[0], path, opset_version=17, do_constant_folding=False,
            input_names=["q","k","v"], output_names=["output"],
            dynamic_axes={"q":{2:"queries"},"k":{2:"keys"},"v":{2:"keys"},"output":{2:"queries"}})
        session = ort.InferenceSession(path,providers=["CPUExecutionProvider"])
        for f,e in zip(fixtures,expected):
            actual = session.run(None,dict(zip(("q","k","v"),[x.numpy() for x in f])))[0]
            np.testing.assert_allclose(actual,e,rtol=0.002,atol=0.0002)
            print("Shaw dynamic parity PASS",f[0].shape,f[1].shape,flush=True)


if __name__ == "__main__":
    main()
