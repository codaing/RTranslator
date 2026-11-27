"""Pinned publicly available Mandarin speech samples; no model weights downloaded."""
import subprocess
from pathlib import Path

BASE = "https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14/resolve/def027084691107096b5ebba69785756d63de6c5/test_wavs/"


def prepare(directory):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    combined=directory/"mandarin.pcm"
    with combined.open("wb") as out:
        for name in ("0.wav","1.wav","2.wav"):
            local=directory/name
            subprocess.run(["curl","--fail","--location","--retry","3","--max-time","120","-o",str(local),BASE+name],check=True)
            raw=subprocess.check_output(["ffmpeg","-v","error","-i",str(local),"-ar","16000","-ac","1","-f","s16le","-"])
            if len(raw)<32000:
                raise ValueError("Speech fixture too short")
            out.write(raw)
    return combined


if __name__=="__main__":
    import sys
    print(prepare(sys.argv[1]))
