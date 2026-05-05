import os
import gradio as gr
import torch
import safetensors.torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
import editdistance
from collections import Counter
from huggingface_hub import login, hf_hub_download

# ── Auth ──────────────────────────────────────────────────────────────────────
hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
if hf_token:
    login(token=hf_token)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL  = "Qwen/Qwen2-VL-7B-Instruct"
ADAPTER_ID  = "shlokchorge2929/visiontex-qwen2vl"
INSTRUCTION = "Write the LaTeX representation for this image."

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading processor...")
processor = AutoProcessor.from_pretrained(BASE_MODEL)

print("Loading base model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

print("Applying LoRA adapter (Unsloth-trained)...")
try:
    adapter_path = hf_hub_download(repo_id=ADAPTER_ID, filename="adapter_model.safetensors")
    adapter_weights = safetensors.torch.load_file(adapter_path)

    lora_A = {}
    lora_B = {}

    for k, v in adapter_weights.items():
        # Adapter keys look like:
        # base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight
        # Model keys look like:
        # model.language_model.layers.0.mlp.down_proj.weight
        # So strip "base_model.model.model." prefix
        clean_k = k.replace("base_model.model.model.", "")

        if ".lora_A.weight" in clean_k:
            base_name = clean_k.replace(".lora_A.weight", "")
            lora_A[base_name] = v
        elif ".lora_B.weight" in clean_k:
            base_name = clean_k.replace(".lora_B.weight", "")
            lora_B[base_name] = v

    print(f"Found {len(lora_A)} lora_A keys, sample: {list(lora_A.keys())[:3]}")

    state = model.state_dict()
    merged = 0
    for name in lora_A:
        w_name = name + ".weight"
        if name in lora_B and w_name in state:
            A = lora_A[name].float()
            B = lora_B[name].float()
            delta = B @ A
            state[w_name] = (state[w_name].float() + delta.to(state[w_name].device)).half()
            merged += 1

    model.load_state_dict(state, strict=False)
    print(f"Merged {merged} LoRA layers ✓")

except Exception as e:
    import traceback
    print(f"LoRA merge error: {e}")
    traceback.print_exc()

model.eval()
print("Ready ✓")

# ── Metrics ───────────────────────────────────────────────────────────────────
def _token_f1(pred, gold):
    pt = pred.strip().split()
    gt = gold.strip().split()
    if not pt or not gt:
        return 0.0
    common = Counter(pt) & Counter(gt)
    tp = sum(common.values())
    pr = tp / len(pt)
    rc = tp / len(gt)
    return round((2 * pr * rc / (pr + rc)) if (pr + rc) else 0.0, 4)

def _cer(pred, gold):
    p, g = pred.strip(), gold.strip()
    return round(min(editdistance.eval(p, g) / max(len(g), 1), 1.0), 4)

# ── Inference ─────────────────────────────────────────────────────────────────
def generate(image: Image.Image, reference: str):
    if image is None:
        return "", "⚠ Please upload an image."
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}]}]
    text     = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs   = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    generated = out[:, inputs["input_ids"].shape[1]:]
    pred = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    if reference.strip():
        f1  = _token_f1(pred, reference)
        cer = _cer(pred, reference)
        metrics = f"Token F1: {f1:.4f}   |   CER: {cer:.4f}   |   {'✓ CER < 0.10' if cer < 0.10 else '✗ CER ≥ 0.10'}"
    else:
        metrics = "Paste ground-truth LaTeX above to see live metrics."
    return pred, metrics

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="VisionTeX") as demo:
    gr.Markdown("## VisionTeX — LaTeX OCR\nFine-tuned Qwen2-VL-7B · Provide ground truth to compute Token F1 and CER live.")
    with gr.Row():
        with gr.Column():
            img_in  = gr.Image(type="pil", label="Input Image")
            ref_in  = gr.Textbox(label="Ground-truth LaTeX (optional)", lines=2)
            btn     = gr.Button("Generate", variant="primary")
        with gr.Column():
            out_tex = gr.Textbox(label="Predicted LaTeX", lines=6)
            out_met = gr.Textbox(label="Metrics", lines=2)
    btn.click(fn=generate, inputs=[img_in, ref_in], outputs=[out_tex, out_met])

if __name__ == "__main__":
    demo.launch()
