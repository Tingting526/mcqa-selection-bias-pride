#!/usr/bin/env python3
"""CPU-only check of every retained inference prompt hash and token length."""
from pathlib import Path
import hashlib
import json
import os
os.environ['USE_TORCH'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'analysis/outputs/thesis_strengthening_20261021'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = json.loads((OUT/'inference_plan.json').read_text())
    items = {(r['condition'], r['id']):r for r in plan['requests']}
    previous_path = OUT/'prompt_serialization_validation.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    for model in plan['models']:
        for phase in ('mmlu', 'chat'):
            run = 'chat_a100' if phase == 'chat' and model == 'mistral-7b-instruct-v0.3' else phase
            directory = OUT/'gpu_results'/model/run
            manifest_path = directory/'manifest.json'
            if not manifest_path.exists(): continue
            manifest = json.loads(manifest_path.read_text())
            if manifest['status'] != 'completed': continue
            key = model+'/'+run
            hashes = dict(manifest_sha256=sha(manifest_path), records_sha256=sha(directory/'records.jsonl'),
                          plan_sha256=sha(OUT/'inference_plan.json'))
            assert manifest['plan_sha256'] == hashes['plan_sha256']
            assert manifest['records_sha256'] == hashes['records_sha256']
            for name,value in manifest['tokenizer_hashes'].items():
                assert sha(directory/'tokenizer'/name) == value
            if key in previous and all(previous[key][k] == v for k,v in hashes.items()):
                continue
            tokenizer = AutoTokenizer.from_pretrained(directory/'tokenizer', use_fast=False,
                add_bos_token=False, add_eos_token=False, trust_remote_code=False, local_files_only=True)
            slots = [tokenizer(prefix+label).input_ids[-1] for prefix in (': ', ':') for label in 'ABCD']
            assert slots == manifest['slot_ids']
            trailing = slots[:4] == slots[4:]
            count, above_cap = 0, 0
            for line in (directory/'records.jsonl').open():
                record = json.loads(line); item = items[record['condition'],record['sample_id']]
                for s in range(4):
                    options = item['options'][s:] + item['options'][:s]
                    subject = ' about '+item['subject'].replace('_',' ') if item['dataset'] == 'mmlu' else ''
                    instruction = 'The following are multiple choice questions'+subject+'. You should directly answer the question by choosing the correct option.'
                    choices = '\n'.join((label+'. '+text).strip() for label,text in zip('ABCD',options))
                    task = instruction+'\n\nQuestion: '+item['question'].strip()+'\nOptions:\n'+choices+'\nAnswer:'
                    native = item['condition'] == 'medical_native_chat'
                    if native:
                        text = tokenizer.apply_chat_template([dict(role='user',content=task)], tokenize=False,
                            add_generation_prompt=True, **plan['models'][model].get('chat_template_kwargs',{}))
                        tokens = tokenizer(text,add_special_tokens=False).input_ids
                    else:
                        text = task + (' ' if trailing else '')
                        tokens = tokenizer(text).input_ids
                    assert hashlib.sha256(text.encode()).hexdigest() == record['prompt_sha256'][s]
                    assert len(tokens) == record['input_tokens_before_crop'][s]
                    count += 1; above_cap += len(tokens) > plan['context_cap']
            previous[key] = dict(**hashes, prompt_hashes_and_lengths_checked=count,
                                 above_cap_scorings=above_cap, candidate_slots_match=True)
            previous_path.write_text(json.dumps(previous,indent=2)+'\n')
            print('Validated prompt serialization:', key, count, flush=True)
    print('Total validated scorings:',sum(x['prompt_hashes_and_lengths_checked'] for x in previous.values()),flush=True)


if __name__ == '__main__': main()
