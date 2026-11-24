"""Explicit reciprocal block swaps, preserving recipient embeddings/head/stops."""
import copy,math
from contextlib import contextmanager,nullcontext
from shared import *
from cliff.experiments.revision_causal import transformer_blocks,_text_config
from cliff.experiments.intervention import parameter_digest

def token_nll(b,ids):
    import torch
    if len(ids)!=128 or len(ids)>b.max_seq_len:raise ValueError('Wrong fixed language span')
    model=b.torch_model();x=torch.tensor([ids],device=model.get_input_embeddings().weight.device)
    with torch.inference_mode():
        logits=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False).logits
        losses=-torch.log_softmax(logits[0,:-1].float(),dim=-1).gather(-1,x[0,1:,None]).squeeze(-1)
        if not torch.isfinite(losses).all():raise ValueError('Nonfinite language loss')
        values=losses.cpu().double().tolist()
    total=math.fsum(values)
    return dict(nll_sum=total,token_count=len(values),nll_per_token=total/len(values),token_nll=values,
                reduction='float32 log-softmax; CPU float64 token sum')

def compatible(r,d):
    a,b=_text_config(r.torch_model()),_text_config(d.torch_model())
    # Stop IDs are recipient generation policy; they are not block computation.
    differences={k:[a.get(k),b.get(k)] for k in set(a)|set(b) if a.get(k)!=b.get(k)}
    if set(differences)-{'eos_token_id','bos_token_id','pad_token_id'}:
        raise ValueError('Block architecture differs: '+str(differences))
    rb,db=transformer_blocks(r.torch_model()),transformer_blocks(d.torch_model())
    if len(rb)!=len(db):raise ValueError('Depth differs')
    for x,y in zip(rb,db):
        if {n:type(m).__qualname__ for n,m in x.named_modules()}!={n:type(m).__qualname__ for n,m in y.named_modules()}:
            raise ValueError('Block module classes differ')
        xs,ys=x.state_dict(),y.state_dict()
        if set(xs)!=set(ys) or any(xs[k].shape!=ys[k].shape or xs[k].dtype!=ys[k].dtype for k in xs):
            raise ValueError('Block tensors differ')
    return dict(architecture_compatible=True,recipient_policy_differences=differences,
                tokenizer_policy='Exact actual prompt and label sequences checked; special-token dictionaries may differ.',
                coordinate_alignment='Assumed shared-base alignment, not established mechanistic equivalence.')

@contextmanager
def swap(r,d,layers):
    blocks=transformer_blocks(r.torch_model());donors=transformer_blocks(d.torch_model())
    original={i:blocks[i] for i in layers}
    try:
        for i in layers:blocks[i]=copy.deepcopy(donors[i])
        yield
    finally:
        for i,b in original.items():blocks[i]=b

def same_tokens(r,d,prompt):
    if r.encode_prompt(prompt)!=d.encode_prompt(prompt) or r.label_encodings(prompt,LABELS)!=d.label_encodings(prompt,LABELS):
        raise ValueError('Actual splice prompt/label tokenization differs')

def main(j):
    cfg=j.cfg; inputs=read(HERE/'registration/inputs.json')
    r=load_model(cfg['model']);d=load_model(cfg['donor']) if cfg['layers'] else r
    compatibility=compatible(r,d);before=parameter_digest(r.torch_model(),trainable=False)
    queries=[Example(**q) for q in inputs[cfg['role']+'_queries']]
    contexts=[c for c in inputs['contexts'] if cfg['role']=='final' or c['draw']==0]
    # Exact self-swap and restoration controls are mandatory for every band job.
    control=render_prompt('ag_news',[Example(**x) for x in contexts[1]['demonstrations']],queries[0].text)
    parent=measure(r,control)
    with swap(r,r,cfg['layers']):self_row=measure(r,control)
    if parent['label_scores']!=self_row['label_scores'] or parent['generation']['raw_text']!=self_row['generation']['raw_text']:
        raise ValueError('Self-splice identity failed')
    with swap(r,d,cfg['layers']):
        for context in contexts:
            for q in queries:
                prompt=render_prompt('ag_news',[Example(**x) for x in context['demonstrations']],q.text)
                same_tokens(r,d,prompt)
                def compute():
                    row=measure(r,prompt)
                    return dict(row,query_id=q.example_id,query_hash=q.text_hash,gold_label=q.label,
                                k=context['k'],draw=context['draw'],context_sha256=content_hash(context),layers=cfg['layers'])
                j.record([context['k'],context['draw'],q.example_id],compute)
        language=[]
        for span in read(HERE/'registration/language_spans.json')['spans']:
            ids=span['token_ids']
            if len(ids)!=128:raise ValueError('Language probe span length changed')
            score=token_nll(r,ids)
            language.append(dict(span_id=span['span_id'],input_token_ids_hash=content_hash(ids),**score))
        atomic(j.path/'language.json',dict(spans=language,corpus='held-out AG News source documents; not a broad language benchmark',
                                         mean_nll=sum(x['nll_sum'] for x in language)/sum(x['token_count'] for x in language)))
        if cfg['role']=='pilot':
            # Diagnostic only: no outcome-based template or checkpoint selection.
            r.template='native'
            native=[]
            for context in contexts[:2]:
                for q in queries:
                    prompt=render_prompt('ag_news',[Example(**x) for x in context['demonstrations']],q.text)
                    native.append(dict(query_id=q.example_id,k=context['k'],**measure(r,prompt)))
            atomic(j.path/'native_low_k_diagnostic.json',dict(records=native,selection_used=False))
            r.template='raw'
    after=parameter_digest(r.torch_model(),trainable=False)
    restored=measure(r,control)
    if before!=after or parent['label_scores']!=restored['label_scores']:raise ValueError('Parent restoration failed')
    j.complete(compatibility=compatibility,self_splice_exact=True,restored_parent_sha256=after,
               language_spans=50,scientific_interpretation='Evaluate high-K change jointly with low-K and language damage; retain null/adverse results.')

if __name__=='__main__':run_entry(main)
