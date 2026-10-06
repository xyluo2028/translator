# RTX 4080 Gemma 4 inference, 2026-10-06

Hardware: NVIDIA GeForce RTX 4080, 16 GB; driver 616.92. Tested the translator's real provider calls with
English-to-Chinese dictionary requests, thinking disabled. Timings below are end-to-end provider calls
after weights were loaded, with seed 42, temperature 0, and spelling correction disabled.

| Configuration | Warm lookup time | Decode rate |
| --- | --- | --- |
| User's original Transformers configuration | Reported 18 s | Not measured |
| E4B, Transformers 5.17.0, bitsandbytes 0.50.2 NF4, BF16, SDPA, full CUDA placement | 21.5–26.9 s before compact output prompt | About 14 tokens/s on a fixed 32-token decode |
| Official E4B QAT Q4_0 GGUF, Ollama 0.35.1, compact JSON, generate endpoint | 1.30–1.81 s | About 130 tokens/s |

Final QAT measurements with `config.gpu.toml`:

| Request | Seconds | Output tokens |
| --- | --- | --- |
| book, first request in this already-loaded session | 1.573 | 168 |
| book, repeat | 1.298 | 164 |
| bank | 1.728 | 220 |
| type | 1.809 | 230 |
| “Hello, how are you?” translation | 0.359 | — |

The running web UI was also checked over HTTP using its normal temperature 0.2 and spelling settings.
An English-to-Chinese `book` dictionary request returned HTTP 200 with dictionary senses/examples in
2.181 seconds (2.111 seconds reported by the provider). Its model list returned no errors. The optimized
UI was started on `http://127.0.0.1:8766` using the GPU profile.

The model is `hf.co/google/gemma-4-E4B-it-qat-q4_0-gguf:latest`, manifest digest
`0694c7304cdf0354dabdc59f81284499bc319447e96b4e9009f8053315826e1f`. Model download includes roughly 5.2 GB
of text weights plus a 991 MB multimodal projector. Its source is
[Google's official QAT GGUF repository](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf).

Ollama's placement check reported `100% GPU`, context 2048, and 43/43 layers offloaded to CUDA.
The runner uses about 3.1 GB of reported VRAM; it also maps embedding weights on CPU. This is a layer
placement metric, not a guarantee that every tensor and every operation is on GPU. CPU still handles
embedding lookups, tokenization, request processing, and JSON parsing. For strict placement of all parameters
and buffers on CUDA, the direct Transformers profile was verified at 8.6 GiB, with the slower timing above.

The initial runtime/model warmup was slow: the first full request in the earlier chat/repair configuration
took 62.5 seconds. The table measures warmed performance; startup and reload latency is additional.
The 2048-token context targets short dictionary requests and needs increasing for long texts. These few
examples validate functionality and latency, not comprehensive translation quality; quantization can alter
outputs, and the QAT and bitsandbytes outputs are not token-identical. Both translation and dictionary mode
were exercised. The original 18-second measurement was reported by the user, not reproduced under a
controlled baseline.

Chat/schema and chat/JSON generation exhausted their 512-token budget and caused a second request. The
generate endpoint completed the same compact dictionary prompt in one request. A compiled bitsandbytes
experiment crashed and was left disabled. An older bitsandbytes version and torchao did not improve the
tested eager decode rate; those experiments are not enabled.

Start the runtime with `bash scripts/serve-gpu.sh`, then run
`.venv/bin/python webui.py --config config.gpu.toml --open`.

## Verification after updating system Ollama

The system service at port 11434 was updated from 0.13.4 to 0.35.1. Its cache contains both the imported
Google QAT model and the existing `gpt-oss:latest` model. The optimized UI on port 8766 was restarted
with `config.gpu.system.toml`, and the temporary project-local Ollama daemon was stopped.

The system service reported 43/43 layers on CUDA, Flash Attention enabled, context 2048, and about
3.1 GB of reported VRAM. The CPU embedding caveat above also applies to this service.
Its first warmup request took 56.106 seconds, including 28.997 seconds reported as model loading.
After warmup, real UI requests at temperature 0.2 with normal spelling settings returned HTTP 200:

| Dictionary request | HTTP response time | Reported provider time |
| --- | --- | --- |
| book | 2.178 s | 2.117 s |
| book, repeat | 1.293 s | 1.291 s |
| bank | 2.157 s | 2.156 s |

For this system-service setup, launch the UI with
`.venv/bin/python webui.py --config config.gpu.system.toml --port 8766 --open`.
