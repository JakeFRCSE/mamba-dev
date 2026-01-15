import os
import sys
import torch

import mamba_ssm.modules.mamba_simple as mamba_simple
from mamba_ssm.utils.state_hooks import InferenceParamsCapture

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.config import load_config
from src.utils.tokenizer_utils import load_tokenizer_and_model


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Force CPU conv1d path; causal_conv1d requires CUDA.
    if device.type == "cpu":
        mamba_simple.causal_conv1d_fn = None
        mamba_simple.causal_conv1d_update = None
    config = load_config()
    config.model.device = str(device)
    tokenizer, model = load_tokenizer_and_model(config)
    model = model.to(device)
    inputs = tokenizer(
        "Hello world. This is a longer prompt to exercise the cached states across a split prefill.",
        return_tensors="pt",
    ).input_ids.to(device)
    gen_steps = 15
    split_idx = max(1, inputs.shape[1] // 2)
    prefill_ids = inputs[:, :split_idx]
    rest_ids = inputs[:, split_idx:]

    capture = InferenceParamsCapture(model).enable()
    with torch.no_grad():
        _ = model(prefill_ids)
    params = capture.build_inference_params()
    capture.disable()
    params.seqlen_offset = prefill_ids.shape[1]
    params.max_seqlen = inputs.shape[1] + gen_steps
    params.max_batch_size = inputs.shape[0]
    if rest_ids.numel() > 0:
        for t in range(rest_ids.shape[1]):
            with torch.no_grad():
                _ = model(rest_ids[:, t:t + 1], inference_params=params)
            params.seqlen_offset += 1

    cur = inputs[:, -1:]
    gen_a = []
    for _ in range(gen_steps):
        with torch.no_grad():
            logits = model(cur, inference_params=params, num_last_tokens=1).logits
        cur = logits.argmax(dim=-1)
        gen_a.append(cur)
        params.seqlen_offset += 1

    with torch.no_grad():
        capture_full = InferenceParamsCapture(model).enable()
        _ = model(inputs)
        full_params = capture_full.build_inference_params()
        capture_full.disable()
    full_params.seqlen_offset = inputs.shape[1]
    full_params.max_seqlen = inputs.shape[1] + gen_steps
    full_params.max_batch_size = inputs.shape[0]
    cur = inputs[:, -1:]
    gen_b = []
    for _ in range(gen_steps):
        with torch.no_grad():
            logits = model(cur, inference_params=full_params, num_last_tokens=1).logits
        cur = logits.argmax(dim=-1)
        gen_b.append(cur)
        full_params.seqlen_offset += 1

    gen_a = torch.cat(gen_a, dim=1)
    gen_b = torch.cat(gen_b, dim=1)
    print("gen_a:", gen_a.tolist())
    print("gen_b:", gen_b.tolist())
    print("match:", torch.equal(gen_a, gen_b))
    print("gen_a_text:", tokenizer.decode(gen_a[0]))
    print("gen_b_text:", tokenizer.decode(gen_b[0]))


if __name__ == "__main__":
    main()
