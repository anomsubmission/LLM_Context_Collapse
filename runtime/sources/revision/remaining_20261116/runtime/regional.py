"""Fresh exact-target regional fits and endpoint evaluations; no historical authentication."""
import gc
from shared import *
from cliff.experiments.intervention import consolidate,resolve_targets,parameter_digest
from cliff.experiments.revision_adapter_eval import load_saved_adapter
from cliff.experiments.revision_causal import transformer_blocks

class RegionalJob(Job):
    def record(self,key,compute):
        p=self.path/'records'/(content_hash(key)+'.json')
        if p.exists():row=read(p)
        else:
            start=time.monotonic();row=compute();row.update(key=key,config_sha256=self.digest,recorded_at=utc(),elapsed_seconds=time.monotonic()-start)
            generation_valid(row['generation'],10);atomic(p,row)
        if row['key']!=key or row['config_sha256']!=self.digest:raise ValueError('Resume provenance differs')
        generation_valid(row['generation'],10)
        if row['prediction']!=parse_label_strict(row['generation']['raw_text'],LABELS):raise ValueError('Strict prediction differs')
        self.rows.append(row);self.update(status='evaluating',completed_records=len(self.rows),expected_records=self.cfg['expected_records'])
        return row

def main(j):
    import torch
    cfg=j.cfg;inputs=read(HERE/'registration/inputs.json');design=read(HERE/'registration/regional_inputs.json')
    b=load_model(cfg['model']);model=b.torch_model();window=range(cfg['start'],cfg['start']+cfg['n_blocks'])
    if len(transformer_blocks(model))!=cfg['depth']:raise ValueError('Model depth differs')
    targets=resolve_targets(model,window,'attn')
    atomic(j.path/'exact_targets_before_training.json',dict(targets=targets,blocks=list(window),model=cfg['model'],
        targeting_source='Prospective explicit indices, checked against loaded modules',component='attn'))
    if cfg['mode']!='parent':
        training=j.path/'training';audit_path=training/'training_audit.json'
        rows=design['calibration'][cfg['model_key']][str(cfg['seed'])]
        if cfg['mode']=='pilot':rows=rows[:4]
        if audit_path.exists():audit=read(audit_path)
        else:
            j.update(status='training',calibration_rows=len(rows),optimizer_updates=cfg['updates'])
            audit=consolidate(b,rows,window,'attn',16,32,2e-4,cfg['epochs'],label_only_loss=False,
                train_seed=2026111600+cfg['seed'],output_dir=training,gradient_accumulation_steps=4,
                expected_optimizer_updates=cfg['updates'])
            # Exact save/reload is part of every adapter job, not only its pilot.
            b._model=None;del model;gc.collect()
            if torch.backends.mps.is_available():torch.mps.empty_cache()
            b=load_model(cfg['model'])
        source=dict(directory=training/'adapter',audit=audit,window=window,configuration={'component':'attn'},manifest_path=audit_path)
        reload_info=load_saved_adapter(b,source)
        atomic(j.path/'exact_reload.json',reload_info)
    b.torch_model().requires_grad_(False).eval()
    role='pilot' if cfg['mode']=='pilot' else 'final'
    queries=[Example(**q) for q in inputs[role+'_queries'][:cfg['n_queries']]]
    contexts=[c for c in design['evaluation_contexts'] if c['k'] in [0,cfg['endpoint_k']] and (role=='final' or c['draw']==0)]
    for c in contexts:
        for q in queries:
            def compute():
                prompt=render_prompt('ag_news',[Example(**x) for x in c['demonstrations']],q.text)
                gen=b.generate_record(prompt,10);generation_valid(gen,10);pred=parse_label_strict(gen['raw_text'],LABELS)
                return dict(query_id=q.example_id,query_hash=q.text_hash,gold_label=q.label,k=c['k'],draw=c['draw'],
                    context_sha256=content_hash(c),prompt_sha256=content_hash(prompt),generation=gen,prediction=pred,
                    correct=pred==q.label,valid_generation=pred is not None,seed=cfg['seed'],band=cfg['band'],
                    readout='strict_greedy_generated_label',model_revision=cfg['model']['revision'])
            j.record([q.example_id,c['k'],c['draw']],compute)
    j.complete(exact_targets=targets,blocks=list(window),independent_training_block=cfg['seed'] if cfg['mode']=='adapter' else None,
               historical_targeting_authenticated=False,readout='strict_greedy_generated_label',
               training_audit_sha256=file_hash(j.path/'training/training_audit.json') if cfg['mode']!='parent' else None)
if __name__=='__main__':
    j=RegionalJob(sys.argv[1])
    try:main(j)
    except BaseException as error:j.fail(error);raise
