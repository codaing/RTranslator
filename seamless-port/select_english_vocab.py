"""Select an English-oriented subset with the exact upstream tokenizer.
No evaluation audio or reference predictions are used to select IDs.
"""
import argparse,json,re,unicodedata as ud
from pathlib import Path

MATH_TEXT = (
    "The square root of two is irrational. Let x approach infinity. "
    "The derivative of sine x is cosine x. Integrate the function over the interval. "
    "A quadratic equation has two roots. The determinant of a matrix is nonzero. "
    "Eigenvalues, eigenvectors, orthogonal basis, diagonalization, characteristic polynomial. "
    "Probability distribution, variance, expectation, conditional probability, Bayes theorem. "
    "Continuity, differentiability, convergence, divergence, limit, epsilon, delta, theta, lambda. "
    "Fourier transform, Laplace transform, Taylor series, binomial theorem, partial derivative. "
    "Vector space, linear independence, rank, nullity, tensor, scalar, gradient, Hessian. "
    "x² + y² = r²; ∫ f(x) dx; ∑ₙ₌₁∞; ∂/∂x; √2; π ≈ 3.14159; α β γ Δ Ω. "
    "One minus three over five; negative zero point six; ten to the power of twelve. "
    "Taiwan, Taipei, Mandarin, Chinese, Euclid, Euler, Gauss, Newton, Leibniz, Cauchy, Riemann."
)

def select(pieces, required, budget):
    if budget<=0:raise ValueError("invalid vocabulary budget")
    required=set(required)
    if any(not 0<=i<len(pieces) for i in required):raise ValueError("required ID outside vocabulary")
    # Keep every single Latin/Greek letter, digit, ASCII symbol, common math
    # symbol and byte fallback. Preserve all numeric pieces if budget permits.
    mandatory=set(required);latin=[];numeric=[]
    for i,piece in enumerate(pieces):
        s=piece.replace("▁","")
        if not s or re.fullmatch(r"<0x[0-9A-Fa-f]{2}>",piece):
            mandatory.add(i);continue
        letters=[c for c in s if c.isalpha()]
        latin_only=all("LATIN" in ud.name(c,"") for c in letters)
        if len(s)==1 and (s.isascii() or s.isnumeric() or ud.category(s)=="Sm" or
                            (s.isalpha() and ("LATIN" in ud.name(s,"") or "GREEK" in ud.name(s,"")))):
            mandatory.add(i);continue
        if not letters and any(c.isnumeric() for c in s):numeric.append(i)
        elif letters and latin_only and not piece.startswith("__"):
            latin.append(i)
        elif not letters and all(c.isascii() or ud.category(c) in ("Sm","Pd","Ps","Pe","Pi","Pf","Po") for c in s):
            numeric.append(i)
    if len(mandatory)>budget:raise ValueError("mandatory tokens exceed budget")
    # SentencePiece's vocabulary order is a heuristic, not English frequency.
    # Reserve up to 15% for numbers/symbols; remaining slots favor Latin pieces.
    chosen=set(mandatory)
    for i in numeric[:max(0,min(budget-len(chosen),budget//7))]:chosen.add(i)
    for i in latin:
        if len(chosen)>=budget:break
        chosen.add(i)
    for i in numeric:
        if len(chosen)>=budget:break
        chosen.add(i)
    return sorted(chosen)

def main(directory,budget):
    from seamless_communication.models.unity import load_unity_text_tokenizer
    directory=Path(directory)
    meta=json.loads((directory/"tokenizer.json").read_text())
    tok=load_unity_text_tokenizer("seamless_streaming_unity")
    size=tok.vocab_info.size
    if size!=len(meta["table"]):raise ValueError("tokenizer size differs from model metadata")
    enc=tok.create_encoder(lang="eng",mode="target")
    if enc.prefix_indices.tolist()!=meta["prefix"]:raise ValueError("English prefix mismatch")
    if tok.model.token_to_index(",")!=meta["comma"]:raise ValueError("comma ID mismatch")
    pieces=[str(tok.model.index_to_token(i)) for i in range(size)]
    required=set(meta["prefix"])|{meta[k] for k in ("eos","pad","unk","comma")}|set(range(4))
    required.update(enc(MATH_TEXT).tolist())
    kept=select(pieces,required,budget)
    result={"original_size":size,"retained_ids":kept,"budget":budget,
            "mandatory_math_ids":len(set(enc(MATH_TEXT).tolist())),
            "policy":"Latin-oriented ID order plus numeric/symbol reserve and independent math text"}
    (directory/"english-vocab.json").write_text(json.dumps(result,separators=(",",":"))+"\n")
    print("retained",len(kept),"of",size,"tokens",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("directory");p.add_argument("--budget",type=int,default=48000)
    a=p.parse_args();main(a.directory,a.budget)
