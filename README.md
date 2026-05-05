# VisionTeX — LaTeX OCR via Fine-Tuned Qwen2-VL-7B

VisionTeX is a mathematical expression recognition system built by fine-tuning Qwen2-VL-7B-Instruct on the LaTeX OCR dataset using Unsloth and LoRA. Given an image of a mathematical expression, the model produces the corresponding LaTeX source code. The project covers the full pipeline — training, evaluation, a Gradio demo, and a merged model checkpoint for direct deployment.

Live demo: https://huggingface.co/spaces/shlokchorge2929/visiontex
<img width="1587" height="833" alt="image" src="https://github.com/user-attachments/assets/1971ed89-65eb-4072-b5e1-04d403d05f44" />

---

## What this project does

The model takes an image containing mathematical notation and outputs LaTeX markup that, when compiled, reproduces that expression. This is useful for digitizing handwritten or printed equations from papers, textbooks, or scanned documents without manual transcription.

The fine-tuning is not a tutorial wrapper. Every design choice is backed by a measured result:

- LoRA rank sweep across r=8, r=16, r=32 with F1 comparison
- Data augmentation pipeline (rotation, Gaussian noise, contrast jitter, blur) with before/after error rates
- Cosine vs linear learning rate scheduler comparison — approximately 2% F1 difference quantified
- Four evaluation metrics: Character Error Rate, Token F1, BLEU-4, and Exact Match

---

## Repositories and how they connect

**This repository — shlokchorge/Fine-Tuning-QWEN2vl**

Contains the training notebook and the Gradio application that runs the demo.

**Adapter checkpoint — shlokchorge2929/visiontex-qwen2vl**

After training, Unsloth saves a LoRA adapter — a small set of weight differences rather than a full model copy. This adapter is roughly a few hundred MB compared to the full model's 14+ GB. The Gradio app (`app.py`) downloads only this adapter at startup, applies it on top of the base Qwen2-VL-7B-Instruct weights by computing `delta = B @ A` for each LoRA layer and adding it to the frozen base weights. This is the merge that happens in the loading section of `app.py`.

**Merged checkpoint — shlokchorge2929/visiontex-qwen2vl-merged**

This is the result of permanently folding the adapter into the base weights. Instead of loading two things and merging at runtime, you load one complete model. The weights are identical in value to what the adapter produces — the difference is purely operational: no merge step, no dependency on the base model checkpoint, faster cold start, and easier deployment. If you want to run inference without building the merge logic yourself, use this checkpoint directly with the standard Transformers pipeline.

The relationship is: training produces the adapter, the adapter applied to the base produces the merged model, and the demo uses the adapter approach so it can be hosted without storing 14 GB on the Space.

---

## File structure

```
Fine-Tuning-QWEN2vl/
│
├── visiontex_train_t4.ipynb    Training notebook. Designed to run on T4 (15 GB)
│                               at MAX_STEPS=500 or A100 40 GB for a full epoch.
│                               Contains the augmentation pipeline, LoRA config,
│                               SFT training loop, evaluation functions, and a
│                               before/after comparison table.
│
├── app.py                      Gradio application. Loads the base model and
│                               downloads the LoRA adapter from HuggingFace,
│                               merges the weights at runtime, and exposes a UI
│                               where you can upload an image, optionally paste
│                               ground-truth LaTeX, and get predictions with
│                               live Token F1 and CER scores.
│
└── requirements.txt            Python dependencies for running app.py locally.
```

---

## How to use

**Running the demo**

The hosted demo is at https://huggingface.co/spaces/shlokchorge2929/visiontex. Upload an image of a mathematical expression, optionally paste the ground-truth LaTeX to compute metrics, and click Generate.

**Running locally**

```bash
git clone https://github.com/shlokchorge/Fine-Tuning-QWEN2vl.git
cd Fine-Tuning-QWEN2vl
pip install -r requirements.txt
python app.py
```

You need a GPU with at least 16 GB of VRAM for local inference. The app will download the base model and adapter automatically on first run. Set `HUGGING_FACE_HUB_TOKEN` in your environment if you run into rate limits.

**Running training**

Open `visiontex_train_t4.ipynb` in a Colab or Kaggle environment with a T4 GPU. The default config is set for T4 (MAX_STEPS=500, LoRA r=16, MAX_SEQ_LEN=1024, batch size 1). For a full training run, set `MAX_STEPS = None` and switch to an A100 40 GB instance.

**Using the merged model directly**

```python
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import torch

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "shlokchorge2929/visiontex-qwen2vl-merged",
    torch_dtype=torch.float16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("shlokchorge2929/visiontex-qwen2vl-merged")
```

This loads the fully merged weights without any adapter merge step.

---

## Why inference is slow and what hardware helps

The base model, Qwen2-VL-7B, has seven billion parameters. Even in float16, the weights alone occupy around 14 GB of GPU memory. Before generating a single token the model must load those weights, encode the image through the vision encoder, and then run an autoregressive decode loop — each token requires a full forward pass through all layers.

On the HuggingFace Space the model runs on a CPU-only or shared GPU instance. CPU inference on a 7B model is roughly 50 to 100 times slower than a dedicated GPU. Even with a T4 (which has 16 GB VRAM), throughput is limited because the T4 has relatively low memory bandwidth and no bfloat16 support, so the model runs in float16 with slower attention.

The hardware that would make a meaningful difference:

- A100 40 GB or 80 GB — the recommended card for this model class. Memory bandwidth is roughly 3x the T4, and bfloat16 is supported natively, which speeds up both training and inference.
- RTX 4090 (24 GB) — a practical consumer alternative. Not as fast as an A100 for training but significantly faster than a T4 for inference.
- H100 — the fastest available option, most relevant if you want to run full-epoch training with larger batch sizes.

Quantization to 4-bit (which the training notebook already uses via Unsloth) reduces VRAM from ~14 GB to around 5-6 GB, making T4 inference possible at the cost of some quality. The hosted Space uses the unquantized adapter merge approach, which requires more memory and runs slower as a result.

---

## Contributing

If you want to help this project run faster and more reliably, hardware access is the primary constraint. Specifically:

- Access to an A100 or H100 instance (Google Cloud, Lambda Labs, RunPod, or similar) to run a full-epoch training run and produce a properly converged checkpoint. The current adapter was trained at MAX_STEPS=500 on a T4, which is a partial run.
- Sponsoring compute credits on any major cloud provider so the Space can be upgraded to a GPU-backed instance and serve inference in reasonable time.
- Contributions to the codebase: quantized inference paths (GPTQ, AWQ), faster attention (Flash Attention 2), or a lighter evaluation server that handles the adapter merge more efficiently.

If you have compute to offer or want to collaborate on any of the above, open an issue in this repository or reach out directly.
