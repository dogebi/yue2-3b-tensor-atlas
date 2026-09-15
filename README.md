# YuE2-3B — Tensor Atlas v4

An offline, single-file WebGL2 architecture atlas for **[m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B)**,
built with the `tensor-atlas-v4-retarget` skill: nisten's Tensor Atlas v4 engine kept whole, the model half
replaced. `index.html` is self-contained — no CDN, no network calls at runtime.

Live: https://yue2-3b-tensor-atlas.netlify.app

## What this model is

YuE2-3B is a music-generation backbone that is literally a **Mixture of Transformers**: each of the 28 layers
carries an autoregressive stack (`self_attn` + `mlp`) for the score and semantic tokens **and** a
non-autoregressive stack (`nar_self_attn` + `nar_mlp`) for the acoustic latents, in the same weights. Two VAE
bridges (`llm2vae` 2048 → 64, `vae2llm` 64 → 2048) connect the language space to the latent space that a
separate decoder (m-a-p/YuE2-Vae, 0.531 GB, not part of these totals) turns into audio. `time_embedder`
conditions the latent stack on the flow-matching timestep, and `latent_pos_embed.pe` is one learned position
table over 24,576 latent frames.

## Measured payloads

Every byte below is the sum of tensor byte ranges read from the published safetensors **headers** over HTTP
Range — no weights were downloaded, and no figure is a shape × bytes estimate. Quantisation sidecars
(`weight_scale`, `weight_global_scale`, `weight_packed`) are folded into the weight they describe, so all three
modes list the same logical tensors.

| mode | repository | shards | tensors | payload |
|---|---|---|---|---|
| BF16 | m-a-p/YuE2-3B | 1 | 628 | 7.2614 GB |
| FP8 | DKmode22/YuE2-3B-FP8 | 1 | 507 | 2.9238 GB |
| NVFP4 | DKmode22/YuE2-3B-NVFP4 | 1 | 703 | 2.3061 GB |

Both quantised builds are community conversions saved under an upstream arch tag; the atlas folds the tensor
layout, and the page says so in Sources. They also **drop** some tensors (the timestep MLP and the two bridges),
which the page states explicitly where it matters instead of pretending a zero-byte tensor is present.

Where the 7.26 GB goes: the untied embedding and head are 1.51 GB (20.8%) of the reference, the latent position
table is 100.7 MB, and the 28 blocks make up the rest at 22 tensors per layer.

## Build

```
python3 measure.py m-a-p/YuE2-3B DKmode22/YuE2-3B-FP8 DKmode22/YuE2-3B-NVFP4   # headers -> hf-*.json
python3 build.py                                                              # gates + preflight -> index.html
python3 preflight.py --selftest                                               # prove the checker can fail
```

`build.py` refuses to write `index.html` unless: the page's declared totals equal the measured payload for all
three modes, the expected strings are present and the original model's residue is gone, the preflight
self-test passes, and the candidate page passes every structural check that applies to it (including
`node --check` on the inline script, `pos()` targets, dynamic id loops, palette keys, the tick guard).

## Layout

```
index.html        the deliverable — open it from disk, it needs nothing
src.html          nisten's Tensor Atlas v4 engine (MIT, (c) 2026 netsin), unmodified
build.py          data half + gates: folds shard headers into logical tensors, then rewrites/patches the engine
panels.py         panel copy and whole-line rewrites (anchored on the engine's own text)
objectcode.py     rewrite helpers: balance-checked line rewrites, literal substitutions
preflight.py      the structural checker (run it after any edit)
measure.py        safetensors header reader (HTTP Range, header only)
hf-*.json         measured headers and the three published configs
```

## Credits

* Engine: nisten, **LLMViz-DeepSeek-V4.1-Flash** (MIT, (c) 2026 netsin) — reused unmodified.
* Architecture facts: the m-a-p/YuE2-3B model card and `config.json`.
* Quantised builds: DKmode22 (FP8, NVFP4) — measured, not redistributed.
* Tooling: the `tensor-atlas-v4-retarget` skill.
