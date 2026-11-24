"""Finite, resumable G1/G2 collection with per-record atomic commits.

Run via the shared accelerator launcher. A final job requires an explicit
source-bound pilot acceptance; completing a pilot never launches final work.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / 'revision/followup_20261116'
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'revision/four_day_revision/runtime'))
from cliff.data import Example, TASKS
from cliff.prompting import render_prompt, parse_label_strict
from cliff.revision_records import content_hash, file_hash, environment_record
from cliff.experiments.revision_behaviour import _calibration, verify_loaded_model
from focused_scoring import checked_score, generation_valid, inherited_accelerator

def utc():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(path.read_text())

def atomic(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)

def validate(config, inputs):
    if config['role'] not in ('pilot','final') or config['ks'] != [0,4,16,32]:
        raise ValueError('Unexpected design')
    n = 20 if config['role'] == 'pilot' else 200
    if config['n_queries'] != n or len(inputs[config['role'] + '_queries']) != n:
        raise ValueError('Wrong query count')
    if config['draws'] != ([0] if config['role'] == 'pilot' else [0,1,2]):
        raise ValueError('Wrong draw count')
    if config['content_free_queries'] != ['N/A','[MASK]','']:
        raise ValueError('Changed calibration protocol')
    final = [Example(**q) for q in inputs['final_queries']]
    pilot = [Example(**q) for q in inputs['pilot_queries']]
    demos = [Example(**q) for c in inputs['contexts'] for q in c['demonstrations']]
    for rows in (final, pilot):
        if len({q.example_id for q in rows}) != len(rows) or len({q.text_hash for q in rows}) != len(rows):
            raise ValueError('Duplicate queries')
    for a,b in ((final,pilot),(final,demos),(pilot,demos)):
        if ({q.example_id for q in a} & {q.example_id for q in b}
                or {q.text_hash for q in a} & {q.text_hash for q in b}):
            raise ValueError('Query/pilot/demo leakage')
    cells = [(c['k'],c['draw']) for c in inputs['contexts']]
    expected = [(0,0)] + [(k,d) for d in [0,1,2] for k in [4,16,32]]
    if cells != expected:
        raise ValueError('Context grid differs, including unique K0')
    for c in inputs['contexts']:
        rows = [Example(**q) for q in c['demonstrations']]
        if len(rows) != c['k'] or len({q.text_hash for q in rows}) != len(rows):
            raise ValueError('Demonstration count or duplication')
        if rows and Counter(q.label_index for q in rows) != {i:c['k']//4 for i in range(4)}:
            raise ValueError('Unbalanced demonstrations')

def evaluate(backend, config, context, query, calibration):
    labels = TASKS['ag_news']['label_names']
    demos = [Example(**q) for q in context['demonstrations']]
    prompt = render_prompt('ag_news', demos, query.text)
    scores, measured = checked_score(backend, prompt, labels)
    generation = backend.generate_record(prompt, config['max_new_tokens'])
    generation_valid(generation, config['max_new_tokens'])
    if (generation.get('input_token_ids_hash') != measured['input_token_ids_hash']
            or generation['input_tokens_before'] != measured['input_tokens_before']):
        raise ValueError('Generation and scoring inputs differ')
    raw = max(labels, key=lambda l:scores[l])
    calibrated = max(labels, key=lambda l:scores[l]-calibration['effective_log_label_prior'][l])
    generated = parse_label_strict(generation['raw_text'], labels)
    return {'query_id': query.example_id, 'query_hash': query.text_hash, 'gold_label': query.label,
            'k': context['k'], 'draw': context['draw'], 'context_sha256': content_hash(context),
            'prompt_sha256': content_hash(prompt), 'label_scores': scores,
            'predictions': {'raw':raw, 'calibrated':calibrated, 'generated':generated},
            'correct': {'raw':raw==query.label, 'calibrated':calibrated==query.label,
                        'generated':generated==query.label}, 'valid_generation':generated is not None,
            'scoring':measured, 'generation':generation}

def record_key(context, query):
    return content_hash([context['k'], context['draw'], query.example_id])

def verify_record(row, config_hash, context, query):
    if (row.get('config_sha256') != config_hash or row.get('query_id') != query.example_id
            or row.get('query_hash') != query.text_hash or row.get('gold_label') != query.label
            or row.get('context_sha256') != content_hash(context)
            or (row.get('k'),row.get('draw')) != (context['k'],context['draw'])):
        raise ValueError('Existing record belongs to different frozen inputs')
    for key in ('scoring','generation'):
        if row[key].get('truncated') or row[key].get('rejected'):
            raise ValueError('Invalid saved measurement')

def summarize(rows):
    cells = defaultdict(list)
    for row in rows:
        cells[row['k']].append(row)
    return {str(k): {'n':len(v), 'accuracy':{r:sum(x['correct'][r] for x in v)/len(v)
                        for r in ['raw','calibrated','generated']},
                    'valid_generation_fraction':sum(x['valid_generation'] for x in v)/len(v),
                    'min_prompt_tokens':min(x['scoring']['input_tokens_before'] for x in v),
                    'max_prompt_tokens':max(x['scoring']['input_tokens_before'] for x in v)}
            for k,v in sorted(cells.items())}

def verify_sources(registration):
    for name,digest in registration['runtime_sources'].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError(f'Frozen source changed: {name}')

def run(slug):
    inherited_accelerator()
    config_path = HERE / 'registration' / f'{slug}.json'
    config = read(config_path)
    inputs_path = HERE / 'registration/inputs.json'
    inputs = read(inputs_path)
    registration = read(HERE / 'registration/runtime_freeze.json')
    if file_hash(inputs_path) != config['inputs_sha256']:
        raise ValueError('Frozen inputs changed')
    config_hash = file_hash(config_path)
    if registration['configs'].get(config_path.name) != config_hash:
        raise ValueError('Config is not registered')
    verify_sources(registration)
    validate(config,inputs)
    if config['role'] == 'final':
        acceptance = read(HERE / 'registration' / f'{slug.removesuffix("_final")}_pilot_acceptance.json')
        pilot_summary = HERE / 'outputs' / slug.replace('_final','_pilot') / 'summary.json'
        if (acceptance.get('accepted') is not True or acceptance.get('final_config_sha256') != config_hash
                or acceptance.get('pilot_summary_sha256') != file_hash(pilot_summary)):
            raise ValueError('Final collection requires exact pilot acceptance')
    out = HERE / 'outputs' / slug
    out.mkdir(parents=True,exist_ok=True)
    (out / 'records').mkdir(exist_ok=True)
    (out / 'calibrations').mkdir(exist_ok=True)
    with (out / 'execution.lock').open('a') as job_lock:
        fcntl.flock(job_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state = {'job_id':config['job_id'], 'started_at':utc(), 'host':platform.node(),
                 'status':'loading', 'config_sha256':config_hash, 'scientific_evidence':False,
                 'role':config['role'], 'completed_records':0}
        atomic(out/'status.json',state)
        started = time.monotonic()
        try:
            from cliff.revision_runtime import configure_cache
            configure_cache(ROOT/'code/.cache/revision')
            os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                              TOKENIZERS_PARALLELISM='false')
            model = config['model']
            # Retain the exact code used on each host alongside raw results.
            for name,digest in registration['runtime_sources'].items():
                target = out/'source_snapshot'/name
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists() and file_hash(target) != digest:
                    raise ValueError('Existing source snapshot differs')
                if not target.exists():
                    shutil.copyfile(ROOT/name,target)
            snapshot = ROOT/'code/.cache/revision/hub'/('models--'+model['id'].replace('/','--'))/'snapshots'/model['revision']
            index = read(snapshot/'model.safetensors.index.json')
            shards = sorted(set(index['weight_map'].values()))
            payloads = {name:file_hash(snapshot/name) for name in shards}
            from cliff.backends import get_backend
            backend = get_backend('hf_local',model['id'], **{k:v for k,v in model.items() if k!='id'})
            resolved = verify_loaded_model(model,backend,'final_test')
            import torch
            backend.torch_model().requires_grad_(False)
            if str(next(backend.torch_model().parameters()).dtype) != 'torch.bfloat16':
                raise ValueError('Loaded dtype differs')
            atomic(out/'provenance.json', {'config':config, 'environment':environment_record(),
                'resolved':resolved, 'checkpoint_shards_sha256':payloads,
                'runtime_freeze_sha256':file_hash(HERE/'registration/runtime_freeze.json')})
            queries = [Example(**q) for q in inputs[config['role']+'_queries']]
            contexts = [c for c in inputs['contexts'] if c['draw'] in config['draws']]
            rows = []
            evaluation_started = time.monotonic()
            for context in contexts:
                context_hash = content_hash(context)
                prior_path = out/'calibrations'/f'{context_hash}.json'
                if prior_path.exists():
                    prior = read(prior_path)
                    if prior['config_sha256'] != config_hash or prior['context_sha256'] != context_hash:
                        raise ValueError('Calibration provenance mismatch')
                else:
                    demos = [Example(**q) for q in context['demonstrations']]
                    prior = _calibration(backend,'ag_news',demos,TASKS['ag_news']['label_names'],
                                         config['content_free_queries'],config['calibration_floor'])
                    prior.update(config_sha256=config_hash,context_sha256=context_hash)
                    atomic(prior_path,prior)
                for query in queries:
                    path = out/'records'/f'{record_key(context,query)}.json'
                    if path.exists():
                        row = read(path)
                    else:
                        before = time.monotonic()
                        row = evaluate(backend,config,context,query,prior)
                        row.update(config_sha256=config_hash, elapsed_seconds=time.monotonic()-before,
                                   recorded_at=utc(), model_id=model['id'],model_revision=model['revision'],
                                   role=config['role'], scientific_evidence=False)
                        atomic(path,row)
                    verify_record(row,config_hash,context,query)
                    rows.append(row)
                    state.update(status='running',completed_records=len(rows),expected_records=len(queries)*len(contexts),
                                 updated_at=utc(),elapsed_seconds=time.monotonic()-started)
                    atomic(out/'status.json',state)
            if len(list((out/'records').glob('*.json'))) != len(rows):
                raise ValueError('Unexpected extra records')
            verify_sources(registration)
            summary = {'role':config['role'], 'config_sha256':config_hash, 'n_records':len(rows),
                       'n_queries':len(queries), 'cells':summarize(rows),
                       'elapsed_seconds':time.monotonic()-started,
                       'evaluation_seconds':time.monotonic()-evaluation_started,
                       'sum_record_seconds':sum(r['elapsed_seconds'] for r in rows),
                       'mps_driver_allocated_bytes_at_end':torch.mps.driver_allocated_memory(),
                       'scientific_evidence':False,
                       'interpretation':'Technical pilot only' if config['role']=='pilot' else 'Complete collection; paired independent analysis pending',
                       'records_sha256':{p.name:file_hash(p) for p in sorted((out/'records').glob('*.json'))}}
            atomic(out/'summary.json',summary)
            state.update(status='complete',ended_at=utc(),summary_sha256=file_hash(out/'summary.json'))
            atomic(out/'status.json',state)
            print(json.dumps({k:summary[k] for k in ['role','n_records','elapsed_seconds','cells']}),flush=True)
        except BaseException as error:
            state.update(status='failed',error_type=type(error).__name__,error=str(error),updated_at=utc())
            atomic(out/'status.json',state)
            raise

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('config',help='registered config basename, e.g. instruct_pilot')
    args=parser.parse_args()
    if not args.config.replace('_','').isalnum():
        parser.error('invalid basename')
    run(args.config)
