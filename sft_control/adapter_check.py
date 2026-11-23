# Inspect a LoRA adapter checkpoint: did the linear-layer LoRA weights actually train?
# Usage:
#   python adapter_check.py output_evaluation_Nemo_shard0of4/checkpoints/checkpoint-50
import os, sys, json
import torch
from safetensors import safe_open

ckpt = sys.argv[1]
cfg = json.load(open(os.path.join(ckpt, 'adapter_config.json')))
r, alpha = cfg['r'], cfg['lora_alpha']
scale = alpha / (r ** 0.5) if cfg.get('use_rslora') else alpha / r
print(f"r={r} alpha={alpha} rslora={cfg.get('use_rslora')} -> scale={scale:.3f}")
print(f"target_modules={cfg.get('target_modules')}")
print(f"modules_to_save={cfg.get('modules_to_save')}\n")

h = safe_open(os.path.join(ckpt, 'adapter_model.safetensors'), 'pt')
keys = list(h.keys())
a_keys = sorted(k for k in keys if 'lora_A' in k)
other = sorted(k for k in keys if 'lora_A' not in k and 'lora_B' not in k)

n_zero_b, rows = 0, []
for i, ka in enumerate(a_keys):
    if i % 50 == 0: print(f'  ...{i}/{len(a_keys)}', flush=True)
    kb = ka.replace('lora_A', 'lora_B')
    if kb not in keys:
        print(f'!! missing {kb}'); continue
    A, B = h.get_tensor(ka).float(), h.get_tensor(kb).float()
    b_norm = B.norm().item()
    n_zero_b += int(b_norm == 0.0)
    if A.dim() == 2 and B.dim() == 2:
        # ||scale*B@A||_F computed with small r x r matrices (fast, no big matmul)
        fro2 = (scale ** 2) * torch.trace((A @ A.T) @ (B.T @ B)).clamp(min=0).item()
        rms = (fro2 / (B.shape[0] * A.shape[1])) ** 0.5   # typical size of one weight change
        rows.append((rms, fro2 ** 0.5, b_norm, ka))
    else:  # embedding-style LoRA
        rows.append((float('nan'), float('nan'), b_norm, ka))

print(f"LoRA modules: {len(a_keys)}, with lora_B exactly zero: {n_zero_b}")
if rows:
    import statistics
    finite = [x for x in rows if x[0] == x[0]]
    print(f"delta rms    median {statistics.median(x[0] for x in finite):.3e}   max {max(x[0] for x in finite):.3e}")
    print(f"lora_B norm   median {statistics.median(x[2] for x in rows):.3e}   max {max(x[2] for x in rows):.3e}")
    print("\n5 modules with largest delta:")
    for d_max, d_norm, b_norm, k in sorted(finite, reverse=True)[:5]:
        print(f"  rms(delta)={d_max:.3e}  ||delta||={d_norm:.3e}  ||B||={b_norm:.3e}  {k}")

print(f"\nother tensors in adapter (e.g. embed/lm_head copies): {len(other)}")
for k in other[:10]:
    print(f"  {k} {tuple(h.get_tensor(k).shape)}")
