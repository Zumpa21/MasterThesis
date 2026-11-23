# Compare two saved models tensor by tensor (no GPU needed).
# Usage:
#   python quant_diff_check.py pretrained_models/Mistral-NeMo-Minitron-Full-merged \
#                              output_evaluation_Nemo_shard0of4/finetuned_model
import os, sys, glob
from collections import defaultdict
import torch
from safetensors import safe_open


def open_dir(d):
    handles, index = [], {}
    for f in sorted(glob.glob(os.path.join(d, '*.safetensors'))):
        h = safe_open(f, 'pt')
        handles.append(h)
        for k in h.keys():
            index[k] = h
    return handles, index


def category(key, t):
    if 'embed_tokens' in key or 'lm_head' in key:
        return 'embed_tokens / lm_head (16-bit)'
    if t.dtype == torch.uint8 and key.endswith('.weight'):
        return '4-bit codes (linear layers)'
    if 'absmax' in key or 'quant_map' in key or 'quant_state' in key:
        return '4-bit scales (absmax etc.)'
    return 'other (norms, etc.)'


def main(dir_a, dir_b):
    _, A = open_dir(dir_a)
    _, B = open_dir(dir_b)
    print(f'A: {dir_a} ({len(A)} tensors)\nB: {dir_b} ({len(B)} tensors)')
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    if only_a or only_b:
        print(f'!! keys only in A: {len(only_a)} {only_a[:5]}')
        print(f'!! keys only in B: {len(only_b)} {only_b[:5]}')

    stats = defaultdict(lambda: dict(n=0, changed_tensors=0, elems=0, changed_elems=0, diff_sq=0.0, ref_sq=0.0))
    top = []
    for k in sorted(set(A) & set(B)):
        a, b = A[k].get_tensor(k), B[k].get_tensor(k)
        s = stats[category(k, a)]
        s['n'] += 1
        if a.shape != b.shape or a.dtype != b.dtype:
            print(f'!! shape/dtype mismatch {k}: {a.shape} {a.dtype} vs {b.shape} {b.dtype}')
            continue
        ne = (a != b).sum().item()
        s['elems'] += a.numel()
        s['changed_elems'] += ne
        s['changed_tensors'] += int(ne > 0)
        if a.is_floating_point():
            af, bf = a.float(), b.float()
            d, r = (af - bf).pow(2).sum().item(), af.pow(2).sum().item()
            s['diff_sq'] += d
            s['ref_sq'] += r
        top.append((ne / max(a.numel(), 1), k))

    print('\n=== summary per category ===')
    for cat, s in stats.items():
        frac = s['changed_elems'] / max(s['elems'], 1)
        line = (f"{cat:34} tensors changed {s['changed_tensors']:>4}/{s['n']:<4} "
                f"elements changed {frac:9.5%}")
        if s['ref_sq'] > 0:
            line += f"   relative L2 change {(s['diff_sq'] / s['ref_sq']) ** 0.5:.6f}"
        print(line)

    print('\n=== 10 most changed tensors (fraction of elements changed) ===')
    for frac, k in sorted(top, reverse=True)[:10]:
        print(f'{frac:9.5%}  {k}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
