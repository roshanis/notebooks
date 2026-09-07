"""Optional GPU model experiment. Importing this module downloads nothing."""

import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.data import METRICS, ROOT, verify_sources
from src.evidence import EvidenceEngine, messages
from src.evaluation import check_splits, score_prediction, summarize


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_datasets(root=ROOT):
    manifest = json.loads((root / 'eval/split_manifest.json').read_text())
    for relative, expected in manifest['sha256'].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen dataset changed: {relative}')
    train = read_jsonl(root / 'data/training_examples.jsonl')
    validation = read_jsonl(root / 'data/validation_examples.jsonl')
    test = read_jsonl(root / 'eval/questions.jsonl')
    check_splits(train, validation, test)
    return train, validation, test


def load_model(root=ROOT):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError('Select a Colab NVIDIA GPU runtime before loading the 4-bit model.')
    config = json.loads((root / 'config/model.json').read_text())
    set_seed(config['seed'])
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                     bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['revision'], trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        config['model_id'], revision=config['revision'], quantization_config=quantization,
        device_map={'': torch.cuda.current_device()}, dtype=compute_dtype,
        trust_remote_code=False, use_safetensors=True, attn_implementation='eager')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model.eval()
    return model, tokenizer, config, time.perf_counter() - start


def generate(model, tokenizer, prompt, config):
    import torch

    inputs = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True,
                                          return_tensors='pt', return_dict=True)
    if inputs['input_ids'].shape[1] > config['max_input_tokens']:
        raise ValueError('Input exceeds the token budget; refusing to silently truncate evidence.')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    if model.device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=config['max_new_tokens'], do_sample=False,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    if model.device.type == 'cuda':
        torch.cuda.synchronize()
    duration = time.perf_counter() - start
    new_tokens = output[0, inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), duration, len(new_tokens)


def build_prompt(question, condition, evidence):
    if condition == 'base_closed_book':
        return [
            {'role': 'system', 'content': 'Answer historical ACA Marketplace questions from your existing knowledge. '
             'Abstain if uncertain. No source documents are supplied, so citations must be an empty list. '
             'Return only JSON with answer (string), citations (list), abstain (boolean), and values (object). '
             'Numeric values must be decimal strings under keys STATE.COLUMN; use postal codes or All for national totals. '
             'Fixed column dictionary: ' + json.dumps({m: v[0] for m, v in METRICS.items()})},
            {'role': 'user', 'content': question},
        ]
    return messages(question, evidence)


def preflight(tokenizer, config, root=ROOT):
    train, validation, test = validate_datasets(root)
    engine = EvidenceEngine.load(root)
    lengths = {'training': [], 'validation': [], 'evaluation': []}
    for name, records in [('training', train), ('validation', validation)]:
        lengths[name] = [len(tokenizer.apply_chat_template(r['messages'], tokenize=True)) for r in records]
    for record in test:
        evidence = engine.prepare(record['question'])
        for condition in ['base_closed_book', 'base_with_evidence']:
            prompt = build_prompt(record['question'], condition, evidence)
            lengths['evaluation'].append(len(tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)))
    result = {name: {'n': len(values), 'max_tokens': max(values),
                     'over_limit': sum(n > config['max_input_tokens'] for n in values)}
              for name, values in lengths.items()}
    if any(item['over_limit'] for item in result.values()):
        raise ValueError(f'Token preflight failed; no generation or training started: {result}')
    return result


def completion_dataset(records, tokenizer, config):
    from datasets import Dataset
    result = []
    for record in records:
        chat = record['messages']
        length = len(tokenizer.apply_chat_template(chat, tokenize=True))
        if length > config['max_input_tokens']:
            raise ValueError(f'Training record {record["id"]} exceeds max length; do not truncate targets.')
        result.append({'prompt': chat[:-1], 'completion': chat[-1:]})
    return Dataset.from_list(result)


def attach_saved_adapter(base_model, adapter_dir, config):
    from peft import PeftModel
    adapter_dir = Path(adapter_dir)
    metadata = json.loads((adapter_dir / 'training_metadata.json').read_text())
    if any(metadata['model'][key] != config[key] for key in ['model_id', 'revision']):
        raise ValueError('Adapter and base-model revision do not match')
    for name, expected in metadata['weight_sha256'].items():
        if hashlib.sha256((adapter_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Saved adapter checksum mismatch')
    model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False, use_safetensors=True)
    model.eval()
    return model


def load_adapter(adapter_dir, root=ROOT):
    model, tokenizer, config, seconds = load_model(root)
    return attach_saved_adapter(model, adapter_dir, config), tokenizer, config, seconds


def run_condition(model, tokenizer, config, condition, output_root, root=ROOT, adapter_dir=None):
    if condition not in {'base_closed_book', 'base_with_evidence', 'adapter_with_evidence'}:
        raise ValueError('Unknown experiment condition')
    if condition == 'adapter_with_evidence' and adapter_dir is None:
        raise ValueError('An adapter directory is required for reproducible adapter evaluation')
    if condition.startswith('base_') and getattr(model, 'peft_config', None):
        raise ValueError('A base-model control cannot run with an attached adapter; load a fresh base model.')
    if condition == 'adapter_with_evidence' and not getattr(model, 'peft_config', None):
        raise ValueError('Adapter evaluation requires a model with an attached adapter.')
    _, _, questions = validate_datasets(root)
    sources = verify_sources(root)
    engine = EvidenceEngine.load(root)
    run_dir = Path(output_root) / (condition + '-' + uuid4().hex[:10])
    run_dir.mkdir(parents=True, exist_ok=False)
    import torch
    metadata = {'condition': condition, 'started_at': datetime.now(timezone.utc).isoformat(),
                'model': config, 'python': platform.python_version(),
                'gpu': torch.cuda.get_device_name(), 'cuda': torch.version.cuda,
                'packages': {p: importlib.metadata.version(p) for p in
                             ['torch', 'transformers', 'peft', 'trl', 'accelerate', 'bitsandbytes', 'datasets']},
                'split_manifest': json.loads((root / 'eval/split_manifest.json').read_text()),
                'source_manifest': sources, 'actual_compute_cost_usd': None,
                'project_hashes': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [*sorted((root / 'src').glob('*.py')),
                                             root / 'data/documents/guidance.json', root / 'data/reporting_periods.json',
                                             root / 'requirements-colab.txt']},
                'cost_note': 'Enter from your compute account; no cost is inferred from runtime duration.'}
    if adapter_dir is not None:
        metadata['adapter'] = {'path': str(adapter_dir), 'sha256': {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(adapter_dir).iterdir() if p.is_file()}}
    (run_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    results = []
    model.eval()
    with (run_dir / 'predictions.jsonl').open('x') as stream:
        for question in questions:
            evidence = engine.prepare(question['question'])
            closed_book = condition == 'base_closed_book'
            prompt = build_prompt(question['question'], condition, evidence)
            try:
                answer, seconds, token_count = generate(model, tokenizer, prompt, config)
                error = None
            except Exception as exception:
                answer, seconds, token_count, error = '', None, None, str(exception)
            result = {'id': question['id'], 'question': question['question'], 'category': question['category'],
                      'condition': condition, 'prompt': prompt, 'raw_output': answer,
                      'generation_seconds': seconds, 'output_tokens': token_count,
                      'numeric_applicable': bool(question['expected_values']),
                      'evidence_status': evidence['status'], 'error': error,
                      'scores': score_prediction(answer, question, [] if closed_book else evidence['citations'])}
            stream.write(json.dumps(result) + '\n')
            stream.flush()
            results.append(result)
            print(f'{condition}: {len(results)}/{len(questions)}' + (' (generation error)' if error else ''), flush=True)
            if error and ('out of memory' in error.lower() or 'CUDA' in error):
                raise RuntimeError(f'GPU experiment stopped; completed rows saved in {run_dir}')
    (run_dir / 'summary.json').write_text(json.dumps(summarize(results), indent=2))
    return run_dir


def train_adapter(model, tokenizer, config, output_root, root=ROOT):
    if getattr(model, 'peft_config', None):
        raise ValueError('Training needs a fresh base model. Restart the model stage before retraining an adapter.')
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    train, validation, _ = validate_datasets(root)
    adapter_dir = Path(output_root) / ('adapter-' + uuid4().hex[:10])
    adapter_dir.mkdir(parents=True, exist_ok=False)

    train_data = completion_dataset(train, tokenizer, config)
    validation_data = completion_dataset(validation, tokenizer, config)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                            target_modules='all-linear', bias='none', task_type='CAUSAL_LM'))
    model.config.use_cache = False
    model.print_trainable_parameters()
    use_bf16 = torch.cuda.is_bf16_supported()
    args = SFTConfig(output_dir=str(adapter_dir / 'checkpoints'), num_train_epochs=1,
                     per_device_train_batch_size=1, per_device_eval_batch_size=1,
                     gradient_accumulation_steps=8, learning_rate=2e-4, warmup_ratio=0.05,
                     max_length=config['max_input_tokens'], completion_only_loss=True,
                     assistant_only_loss=False, packing=False, gradient_checkpointing=True,
                     bf16=use_bf16, fp16=not use_bf16, eval_strategy='epoch', save_strategy='epoch',
                     logging_steps=5, report_to='none', push_to_hub=False,
                     seed=config['seed'], data_seed=config['seed'], optim='adamw_torch',
                     eos_token=tokenizer.eos_token)
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_data,
                         eval_dataset=validation_data, processing_class=tokenizer)
    # Verify that prompt labels are masked, while some completion labels are supervised.
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch['labels']
    if not torch.any(labels == -100) or not torch.any(labels != -100):
        raise RuntimeError('Completion-only label masking failed; training was not started.')
    started = time.perf_counter()
    result = trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    model.config.use_cache = True
    model.eval()
    (adapter_dir / 'training_metadata.json').write_text(json.dumps({
        'model': config, 'train_examples': len(train), 'validation_examples': len(validation),
        'seconds': time.perf_counter() - started, 'metrics': result.metrics,
        'weight_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in adapter_dir.glob('*.safetensors')},
        'actual_compute_cost_usd': None, 'split_manifest': json.loads((root / 'eval/split_manifest.json').read_text())}, indent=2))
    return model, adapter_dir
