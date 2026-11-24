"""Same-model query-span residual transport with exact-length unrelated controls."""
from shared import *
from cliff.experiments.revision_causal import (prompt_token_roles,align_query_positions,capture_states,
                                             PrefillPatch,score_labels_with_intervention,transformer_blocks)

def patch_measure(b,roles,states,positions):
    def factory(ids):return PrefillPatch(b.torch_model(),states,positions,ids)
    scores=score_labels_with_intervention(b,roles['prompt'],LABELS,factory,required_prompt_positions=positions)
    with factory(roles['input_token_ids']):generation=b.generate_record(roles['prompt'],10)
    generation_valid(generation,10)
    meta=dict(input_tokens_before=len(roles['input_token_ids']),input_token_ids_hash=generation['input_token_ids_hash'],
              truncated=False,rejected=False,implementation='full_forward_with_prefill_only_query_span_patch')
    return dict(label_scores=scores,prediction=max(LABELS,key=scores.get),generated_label=parse_label_strict(generation['raw_text'],LABELS),
                scoring=meta,generation=generation,prompt_sha256=content_hash(roles['prompt']))

def main(j):
    cfg=j.cfg;inputs=read(HERE/'registration/inputs.json');b=load_model(cfg['model'])
    queries=[Example(**q) for q in inputs[cfg['role']+'_queries']]
    donors=read(HERE/'registration/patch_donors.json')[cfg['role']]
    contexts={(c['k'],c['draw']):[Example(**x) for x in c['demonstrations']] for c in inputs['contexts']}
    model=b.torch_model();blocks=transformer_blocks(model)
    hooks_before=[(len(m._forward_hooks),len(m._forward_pre_hooks)) for m in model.modules()]
    self_max=0.0
    for draw in ([0] if cfg['role']=='pilot' else [0,1,2]):
        for q in queries:
            roles={k:prompt_token_roles(b,'ag_news',contexts[k,draw],q.text) for k in [4,16]}
            def base_compute(k):
                return dict(measure(b,roles[k]['prompt']),query_id=q.example_id,query_hash=q.text_hash,gold_label=q.label,
                            k=k,draw=draw,mode='baseline',layer=None)
            baseline={k:j.record([q.example_id,draw,'baseline',k],lambda k=k:base_compute(k)) for k in [4,16]}
            other=Example(**donors[q.example_id])
            unrelated=prompt_token_roles(b,'ag_news',contexts[4,draw],other.text)
            independent=prompt_token_roles(b,'ag_news',contexts[4,(draw+1)%3],q.text)
            for mode,donor,recipient in [('same_query',roles[4],roles[16]),('unrelated_query',unrelated,roles[16]),
                                         ('self',roles[16],roles[16]),('healthy_control',independent,roles[4])]:
                if mode=='unrelated_query':
                    dp,rp=donor['query_positions'],recipient['query_positions']
                    if len(dp)!=len(rp) or other.text_hash==q.text_hash:raise ValueError('Invalid unrelated span control')
                    alignment=dict(policy='Equal-length query spans, positional one-to-one; donor chosen using token count and identity only.',
                                   donor_query_id=other.example_id,donor_query_hash=other.text_hash,used_gold_labels=False)
                else:dp,rp,alignment=align_query_positions(donor,recipient,'query_span')
                states,provenance=capture_states(b,donor['prompt'],[7,15,31],dp,cache_dir=j.path/'donor_cache')
                for layer in [7,15,31]:
                    def compute():
                        return dict(patch_measure(b,recipient,{layer:states[layer]},rp),query_id=q.example_id,
                                    query_hash=q.text_hash,gold_label=q.label,k=4 if mode=='healthy_control' else 16,
                                    draw=draw,mode=mode,layer=layer,alignment=alignment,donor=provenance,
                                    recipient_query_positions=rp,donor_query_positions=dp)
                    row=j.record([q.example_id,draw,mode,layer],compute)
                    if mode=='self':
                        delta=max(abs(row['label_scores'][l]-baseline[16]['label_scores'][l]) for l in LABELS)
                        self_max=max(self_max,delta)
                        if delta>1e-5 or row['generation']['raw_text']!=baseline[16]['generation']['raw_text']:
                            raise ValueError('Self-patch identity failed')
            if hooks_before!=[(len(m._forward_hooks),len(m._forward_pre_hooks)) for m in model.modules()]:
                raise ValueError('Intervention hooks leaked')
    j.complete(self_patch_max_abs_difference=self_max,hook_cleanup_verified=True,
               unrelated_donor_policy='Exact token-count match, labels unused',prefill_only=True)
if __name__=='__main__':run_entry(main)
