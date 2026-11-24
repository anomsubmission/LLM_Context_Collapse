from shared import *
def main(j):
    cfg=j.cfg;b=load_model(cfg['model']);inputs=read(HERE/'registration/inputs.json')
    queries=[Example(**q) for q in inputs[cfg['role']+'_queries']]
    design=read(HERE/'registration/length_contexts.json')
    for c in design['contexts']:
        if cfg['role']=='pilot' and c['draw']!=0:continue
        for q in queries:
            def compute():
                prompt=render_prompt('ag_news',[Example(**x) for x in c['demonstrations']],q.text)
                target=len(b.encode_prompt(render_prompt('ag_news',[],q.text)))+c['target_extra_tokens']
                row=measure(b,prompt)
                if abs(row['scoring']['input_tokens_before']-target)>design['tolerance_tokens']:raise ValueError('Token matching failed')
                return dict(row,query_id=q.example_id,query_hash=q.text_hash,gold_label=q.label,k=c['k'],draw=c['draw'],
                            budget=c['budget'],target_prompt_tokens=target,context_sha256=content_hash(c))
            j.record([q.example_id,c['draw'],c['k'],c['budget']],compute)
    j.complete(token_matching_verified=True,budget_extra_tokens=design['budget_extra_tokens'],tolerance_tokens=design['tolerance_tokens'])
if __name__=='__main__':run_entry(main)
