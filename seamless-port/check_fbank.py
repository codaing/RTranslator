"""Compare the actual Java streaming frontend against Meta's fairseq2 converter."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from fairseq2.data.audio import WaveformToFbankConverter

raw=Path(sys.argv[1])
features=Path(sys.argv[2])
report=Path(sys.argv[3])
pcm=np.fromfile(raw,dtype="<i2").astype(np.float32)/32768
reference=WaveformToFbankConverter(num_mel_bins=80,waveform_scale=32768,
    standardize=False,device=torch.device("cpu"),dtype=torch.float32)(
    {"waveform":torch.from_numpy(pcm).unsqueeze(0),"sample_rate":16000})["fbank"].numpy()
java=np.fromfile(features,dtype="<f4").reshape(-1,80)
if java.shape!=reference.shape or not np.isfinite(java).all():
    raise RuntimeError(f"frontend shape/finite failure: {java.shape} vs {reference.shape}")
error=java-reference
metrics={"frames":len(java),"mae":float(np.abs(error).mean()),
    "rmse":float(np.sqrt(np.mean(error**2))),"p99_absolute":float(np.percentile(np.abs(error),99)),
    "max_absolute":float(np.abs(error).max())}
# Initial tolerance is stated explicitly, not a claim of exact equivalence.
metrics["parity"]="PASS" if metrics["mae"]<=0.05 and metrics["p99_absolute"]<=0.25 else "FAIL"
report.write_text(json.dumps(metrics,indent=2)+"\n")
print(metrics,flush=True)
if metrics["parity"]!="PASS":
    raise RuntimeError("Java frontend exceeds initial log-mel error tolerance")
