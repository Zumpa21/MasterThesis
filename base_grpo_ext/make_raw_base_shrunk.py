# Make the starting model for experiments 4 and 5: the RAW base model with the SAME vocabulary
# reduction the ARChitects apply before Stage-1 training (run_finetuning_Nemo-full.py), but NO training.
# So raw vs Stage-1 differ only by the Stage-1 SFT, not by tokenizer, grid format or memory use.
# Run it inside base_grpo_ext/ and inside base_grpo_int/ (needs a GPU, ~5 min each):  python make_raw_base_shrunk.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
from model_tools import load_unsloth_4bit, keep_single_char_tokens, save_model_and_tokenizer

BASE = 'nvidia/Mistral-NeMo-Minitron-8B-Base'
# saved in pretrained_models/ of the folder you run it from (base_grpo_ext or base_grpo_int)
OUT = os.path.join(os.getcwd(), 'pretrained_models', 'Mistral-NeMo-Minitron-Raw-shrunk')
# Stage-1 model, only READ here to check that the vocabulary is identical
STAGE1 = '/path/to/MasterThesis/sft_ttrl_int_fix/pretrained_models/Mistral-NeMo-Minitron-Full-merged'

if os.path.exists(OUT):
    raise SystemExit(f'{OUT} already exists, delete it first if you want to rebuild it')

model, tokenizer = load_unsloth_4bit(BASE)
keep_tok = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=') + tokenizer.tokenize('\n')
keep_single_char_tokens(model, tokenizer, keep=keep_tok, remove_unk=True)   # identical call to the ARChitects script
save_model_and_tokenizer(OUT, model, tokenizer)
print('saved', OUT)

# checks: reloads, and has exactly the same token -> id mapping as the Stage-1 model
del model
from transformers import AutoTokenizer
m2, t2 = load_unsloth_4bit(OUT)
t1 = AutoTokenizer.from_pretrained(STAGE1, local_files_only=True)
print('vocab size raw-shrunk:', len(t2), ' Stage-1:', len(t1))
print('embedding shape:', tuple(m2.get_input_embeddings().weight.shape))
print('same vocabulary as Stage-1:', t2.get_vocab() == t1.get_vocab())
sample = 'I012\n345\n+/-=O678\n'
print('same tokenization as Stage-1:', t2(sample)['input_ids'] == t1(sample)['input_ids'])
