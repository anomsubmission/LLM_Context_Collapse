"""Source-bound utilities for the remaining November follow-up jobs."""
import json, os, sys, time, platform, math, fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=ROOT/'revision/remaining_20261116'
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(0,str(ROOT/'revision/followup_20261116/runtime'))
from common_readout import utc, read, atomic
from cliff.revision_records import content_hash, file_hash, environment_record
from cliff.data import Example,TASKS
from cliff.prompting import render_prompt,parse_label_strict
from focused_scoring import checked_score,generation_valid,inherited_accelerator
LABELS=TASKS['ag_news']['label_names']

def load_model(spec):
    from cliff.revision_runtime import configure_cache
    from cliff.backends import get_backend
    from cliff.experiments.revision_behaviour import verify_loaded_model
    configure_cache(ROOT/'code/.cache/revision')
    os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    b=get_backend('hf_local',spec['id'],**{k:v for k,v in spec.items() if k!='id'})
    verify_loaded_model(spec,b,'final_test')
    b.torch_model().requires_grad_(False).eval()
    return b

def measure(b,prompt,max_new_tokens=10):
    scores,meta=checked_score(b,prompt,LABELS)
    generation=b.generate_record(prompt,max_new_tokens)
    generation_valid(generation,max_new_tokens)
    if generation['input_token_ids_hash']!=meta['input_token_ids_hash']:
        raise ValueError('Likelihood/generation token identity differs')
    return dict(label_scores=scores,scoring=meta,generation=generation,
                prediction=max(LABELS,key=scores.get),generated_label=parse_label_strict(generation['raw_text'],LABELS),
                prompt_sha256=content_hash(prompt))

def check_measure(row):
    if set(row['label_scores'])!=set(LABELS) or not all(math.isfinite(x) for x in row['label_scores'].values()):
        raise ValueError('Incomplete/nonfinite label scores')
    if row['prediction']!=max(LABELS,key=row['label_scores'].get):raise ValueError('Prediction differs')
    for k in ('scoring','generation'):
        if row[k].get('truncated') or row[k].get('rejected'):raise ValueError('Truncation/overflow')

class Job:
    def __init__(self,slug):
        inherited_accelerator()
        self.slug=slug; self.path=HERE/'outputs'/slug;self.path.mkdir(parents=True,exist_ok=True)
        (self.path/'records').mkdir(exist_ok=True)
        self.lock=(self.path/'execution.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.cfg=read(HERE/'registration'/f'{slug}.json');self.digest=content_hash(self.cfg)
        self.freeze=read(HERE/'registration'/f'{slug}.freeze.json')
        self.verify()
        self.state=dict(job_id=slug,status='loading',host=platform.node(),pid=os.getpid(),started_at=utc(),
                        config_sha256=self.digest,completed_records=0,scientific_evidence=False)
        self.started=time.monotonic();self.rows=[];self.update()
        atomic(self.path/'provenance.json',dict(config=self.cfg,freeze=self.freeze,environment=environment_record()))
    def verify(self):
        if self.freeze['config_content_sha256']!=self.digest:raise ValueError('Config changed')
        for name,h in self.freeze['files'].items():
            if file_hash(ROOT/name)!=h:raise ValueError('Frozen artifact changed: '+name)
        if self.cfg.get('requires'):
            for slug in self.cfg['requires']:
                s=read(HERE/'outputs'/slug/'status.json')
                if s['status']!='complete':raise ValueError('Prerequisite incomplete: '+slug)
                summary=read(HERE/'outputs'/slug/'summary.json')
                if not summary.get('technical_checks_passed'):raise ValueError('Prerequisite checks failed')
    def update(self,**kw):
        self.state.update(kw,updated_at=utc(),elapsed_seconds=time.monotonic()-self.started)
        atomic(self.path/'status.json',self.state)
    def record(self,key,compute):
        p=self.path/'records'/(content_hash(key)+'.json')
        if p.exists():r=read(p)
        else:
            start=time.monotonic();r=compute();r.update(key=key,config_sha256=self.digest,recorded_at=utc(),elapsed_seconds=time.monotonic()-start)
            check_measure(r);atomic(p,r)
        if r['key']!=key or r['config_sha256']!=self.digest:raise ValueError('Resume identity mismatch')
        check_measure(r);self.rows.append(r)
        self.update(status='running',completed_records=len(self.rows),expected_records=self.cfg['expected_records'])
        return r
    def complete(self,**extra):
        self.verify()
        if len(self.rows)!=self.cfg['expected_records']:raise ValueError('Record count differs')
        summary=dict(n_records=len(self.rows),technical_checks_passed=True,config_sha256=self.digest,
                     elapsed_seconds=time.monotonic()-self.started,completed_at=utc(),**extra)
        summary['records_sha256']={p.name:file_hash(p) for p in (self.path/'records').glob('*.json')}
        atomic(self.path/'summary.json',summary);self.update(status='complete',ended_at=utc())
    def fail(self,e):self.update(status='failed',error_type=type(e).__name__,error=str(e))

def run_entry(fn):
    j=Job(sys.argv[1])
    try:fn(j)
    except BaseException as e:j.fail(e);raise
