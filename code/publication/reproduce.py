"""Recompute manuscript evidence from exported measurements, with no model downloads.

Run: python -m publication.reproduce --out publication/generated
The bundled manifest pins inputs. Every computed contrast is checked against the
independent analysis; null/adverse results and unavailable panels are explicit.
"""
from pathlib import Path
import argparse,csv,gzip,hashlib,json,os
from collections import defaultdict,Counter
import numpy as np
DATA=Path(__file__).resolve().parent/'data'
LABELS=['World','Sports','Business','Sci/Tech']
MODELS=[('llama31','Llama-3.1',16),('deepseek7b','DeepSeek-7B',16),('mistral7b','Mistral',32),('internlm25','InternLM',64)]

def read(p):return json.loads(p.read_text())
def rows(p):
 with p.open() as f:return list(csv.DictReader(f))
def write_rows(p,rs):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def interval(a):
 a=np.asarray(a,dtype=float);nr,nd,nq=a.shape
 if not np.isfinite(a).all():raise ValueError('Incomplete paired observations')
 rng=np.random.default_rng(20261116)
 rw=rng.multinomial(nr,[1/nr]*nr,size=5000)/nr
 dw=rng.multinomial(nd,[1/nd]*nd,size=5000)/nd
 qw=rng.multinomial(nq,[1/nq]*nq,size=5000)/nq
 z=np.einsum('br,bd,bq,rdq->b',rw,dw,qw,a,optimize=True)
 lo,hi=np.quantile(z,[.025,.975])
 return dict(estimate=float(a.mean()),ci95_low=float(lo),ci95_high=float(hi),n_training_blocks=nr,n_draws=nd,n_queries=nq)
def verify_inputs(data):
 manifest=read(data/'manifest.json')
 for name,digest in manifest['files'].items():
  if hashlib.sha256((data/name).read_bytes()).hexdigest()!=digest:raise ValueError('Input hash mismatch: '+name)
 if (data/'adapter_audit.json').exists():verify_regional_audits(data)
 return manifest

def verify_regional_audits(data):
 summaries=read(data/'adapter_audit.json')
 with gzip.open(data/'regional_training_audits.json.gz','rt') as f:full=json.load(f)
 with gzip.open(data/'protocol_companions.json.gz','rt') as f:protocols=json.load(f)
 design=protocols['revision/remaining_20261116/registration/regional_inputs.json']['calibration']
 if len(summaries)!=18 or {r['job'] for r in summaries}!=set(full):raise ValueError('Regional audit inventory differs')
 for r in summaries:
  bundle=full[r['job']];audit=bundle['training_audit'];reload=bundle['exact_reload']
  n=len(design[r['model']][str(r['seed'])])
  if r['calibration_rows']!=n or len(audit['encoding']['sequence_lengths'])!=n:
   raise ValueError('Calibration row count differs: '+r['job'])
  if r['optimizer_updates']!=n//4*2 or audit['optimizer_updates']!=r['optimizer_updates']:
   raise ValueError('Optimizer update count differs: '+r['job'])
  if (audit['adapted_blocks']!=r['blocks'] or audit['trainable_parameters']!=r['trainable_parameters']
      or reload['registered_targets']!=audit['registered_targets']
      or not audit['frozen_weights_unchanged'] or not audit['gradient_audit']['finite']
      or not audit['gradient_audit']['nonzero_observed']
      or audit['initial_adapter_sha256']==audit['adapter_sha256']
      or bundle['original_training_audit_sha256']!=r['training_audit_sha256']):
   raise ValueError('Regional execution audit differs: '+r['job'])
  check=reload['saved_reload_audit']
  if (not check['passed'] or check['exact_adapter_parameter_sha256']!=audit['adapter_sha256']
      or check['exact_frozen_base_parameter_sha256']!=audit['frozen_sha256']):
   raise ValueError('Exact saved reload differs: '+r['job'])

def load_arrays(data):
 table=defaultdict(dict);confusions=Counter();n=0;job_counts=Counter()
 orders=read(data/'query_orders.json')
 configs=read(data/'configs.json') if (data/'configs.json').exists() else {}
 allowed={k:set(v) for k,v in orders.items()}
 with gzip.open(data/'measurements.jsonl.gz','rt') as f:
  for line in f:
   r=json.loads(line);n+=1
   campaign='followup_20261116' if r['family']=='G12' else 'remaining_20261116'
   if r['query'] not in allowed[campaign]:raise ValueError('Unknown query identity')
   if configs:
    cfg=configs.get(r['job'])
    if cfg is None or cfg.get('role')!='final':raise ValueError('Unregistered final job')
    if r['query'] not in orders[campaign][:cfg.get('n_queries',200)]:raise ValueError('Query outside registered panel')
    job_counts[r['job']]+=1
   text=r['text'].strip().rstrip('.').strip().casefold()
   generated=next((l for l in LABELS if l.casefold()==text),None)
   preds={'generated':generated};outcomes={'generated':float(generated==r['gold']),'valid':float(generated is not None)}
   if 'scores' in r:
    scores=np.asarray(r['scores']);assert np.isfinite(scores).all()
    preds['raw']=LABELS[int(scores.argmax())];outcomes['raw']=float(preds['raw']==r['gold'])
   if 'logprior' in r:
    preds['calibrated']=LABELS[int((scores-np.asarray(r['logprior'])).argmax())];outcomes['calibrated']=float(preds['calibrated']==r['gold'])
   for mode,val in outcomes.items():
    key=(r['family'],r['condition'],mode);identity=(r['seed'],r['draw'],r['query'])
    if identity in table[key]:raise ValueError('Duplicate observation')
    table[key][identity]=val
   for mode,p in preds.items():confusions[r['family'],r['condition'],mode,r['gold'],p or '<invalid>']+=1
 if configs:
  for job,cfg in configs.items():
   expected=cfg.get('expected_records')
   if expected is None:expected=cfg['n_queries']*sum(1 if k==0 else len(cfg['draws']) for k in cfg['ks'])
   if job_counts[job]!=expected:raise ValueError('Final job record count mismatch: '+job)
 arrays={}
 for key,values in table.items():
  order=orders['followup_20261116' if key[0]=='G12' else 'remaining_20261116']
  seeds=sorted({x[0] for x in values});draws=sorted({x[1] for x in values});qs=[q for q in order if q in {x[2] for x in values}]
  a=np.array([[[values.get((s,d,q),float('nan')) for q in qs] for d in draws] for s in seeds])
  if not np.isfinite(a).all():raise ValueError('Incomplete paired panel '+str(key))
  arrays[key]=a
 return arrays,n,confusions

def remaining_contrasts(arrays,data):
 reports=[]
 # Match the registered analysis's shared zero-shot draw axis, without changing N.
 a={k:np.repeat(v,3,axis=1) if v.shape[1]==1 else v for k,v in arrays.items() if k[0]!='G12'}
 for key in sorted(a):
  summary=interval(a[key])
  # The repeated axis preserves the registered paired bootstrap calculation;
  # metadata must describe observed draws, not computational broadcast copies.
  summary['n_draws']=arrays[key].shape[1]
  reports.append(dict(family=key[0],condition=key[1],readout=key[2],contrast='mean',**summary))
 def contrast(f,name,r,terms):
  keys=[(f,c,r) for w,c in terms]
  if not all(k in a for k in keys):return
  reports.append(dict(family=f,condition=name,readout=r,contrast='paired_difference',**interval(sum(w*a[f,c,r] for w,c in terms))))
 for r in ['raw','generated','valid']:
  for k in [4,16]:contrast('G3',f'length_effect_K{k}',r,[(1,f'K{k}_budget1'),(-1,f'K{k}_budget0')])
  for b in [0,1]:contrast('G3',f'count_effect_budget{b}',r,[(1,f'K16_budget{b}'),(-1,f'K4_budget{b}')])
  contrast('G3','count_by_length_interaction',r,[(1,'K16_budget1'),(-1,'K16_budget0'),(-1,'K4_budget1'),(1,'K4_budget0')])
  for layer in [7,15,31]:
   same=f'same_query_layer{layer}_K16';other=f'unrelated_query_layer{layer}_K16';selfc=f'self_layer{layer}_K16';healthy=f'healthy_control_layer{layer}_K4'
   contrast('G4',f'layer{layer}_same_minus_unrelated',r,[(1,same),(-1,other)])
   contrast('G4',f'layer{layer}_same_minus_self',r,[(1,same),(-1,selfc)])
   contrast('G4',f'layer{layer}_rescue_minus_healthy_effect',r,[(1,same),(-1,selfc),(-1,healthy),(1,'baseline_layerNone_K4')])
  for recipient in ['instruct','hermes']:
   for band in ['early','middle','late']:
    h=f'G5_{recipient}_{band}';p=f'G5_{recipient}_parent'
    for k in [0,4,16,32]:contrast('G5',f'{h}_minus_parent_K{k}',r,[(1,f'{h}_K{k}'),(-1,f'{p}_K{k}')])
    for k in [16,32]:contrast('G5',f'{h}_difference_in_K{k}_minus_K4',r,[(1,f'{h}_K{k}'),(-1,f'{h}_K4'),(-1,f'{p}_K{k}'),(1,f'{p}_K4')])
 for model,_,k in MODELS:
  for band in ['early','middle','late']:
   for r in ['generated','valid']:
    for endpoint in [0,k]:contrast('G6',f'{model}_{band}_minus_parent_K{endpoint}',r,[(1,f'{model}_{band}_K{endpoint}'),(-1,f'{model}_parent_K{endpoint}')])
    contrast('G6',f'{model}_{band}_difference_in_decline',r,[(1,f'{model}_{band}_K{k}'),(-1,f'{model}_{band}_K0'),(-1,f'{model}_parent_K{k}'),(1,f'{model}_parent_K0')])
 lang=read(data/'language.json')
 for recipient in ['instruct','hermes']:
  base=lang[f'G5_{recipient}_parent_final']
  for band in ['early','middle','late']:
   job=f'G5_{recipient}_{band}_final';rs=lang[job]
   assert [(x['span_id'],x['input_token_ids_hash']) for x in rs]==[(x['span_id'],x['input_token_ids_hash']) for x in base]
   reports.append(dict(family='G5',condition=job,readout='news_domain_language_nll',contrast='hybrid_minus_parent',**interval(np.array([x['nll_per_token']-b['nll_per_token'] for x,b in zip(rs,base)])[None,None,:])))
 return reports

def checkpoint_contrasts(arrays):
 models=['base','instruct','hermes','tulu_sft','tulu_dpo','tulu_final'];modes=['raw','calibrated','generated'];ks=[0,4,16,32]
 a=np.array([[[(np.repeat(arrays['G12',f'{m}_K{k}',r],3,axis=1) if k==0 else arrays['G12',f'{m}_K{k}',r])[0] for k in ks] for r in modes] for m in models])
 rng=np.random.default_rng(20261116);boot=np.empty((5000,6,3,4))
 for i in range(5000):
  qi=rng.integers(0,200,200);di=rng.integers(0,3,3)
  boot[i]=np.take(np.take(a,di,axis=3),qi,axis=4).mean(axis=(3,4))
 means=a.mean(axis=(3,4));out=[]
 def append(m,r,c,val,samples,observed_draws=3):
  lo,hi=np.quantile(samples,[.025,.975]);out.append(dict(model=m,readout=r,contrast=c,estimate=float(val),ci95_low=float(lo),ci95_high=float(hi),n_queries=200,n_demonstration_draws=observed_draws))
 for mi,m in enumerate(models):
  for ri,r in enumerate(modes):
   for ki,k in enumerate(ks):append(m,r,f'accuracy_K{k}',means[mi,ri,ki],boot[:,mi,ri,ki],observed_draws=1 if k==0 else 3)
   for ki,k in [(2,16),(3,32)]:append(m,r,f'K{k}_minus_K4',means[mi,ri,ki]-means[mi,ri,1],boot[:,mi,ri,ki]-boot[:,mi,ri,1])
 for x,y in [(0,1),(0,2),(1,2),(3,4),(4,5),(3,5)]:
  for ri,r in enumerate(modes):
   for ki,k in [(2,16),(3,32)]:append(f'{models[y]}_minus_{models[x]}',r,f'difference_in_K{k}_minus_K4',(means[y,ri,ki]-means[y,ri,1])-(means[x,ri,ki]-means[x,ri,1]),(boot[:,y,ri,ki]-boot[:,y,ri,1])-(boot[:,x,ri,ki]-boot[:,x,ri,1]))
 return out

def regional_contrasts(a):
 out=[]
 for m,_,k in MODELS:
  for l,r in [('early','late'),('middle','late'),('early','middle')]:
   for mode in ['generated','valid']:
    diff={v:a['G6',f'{m}_{l}_K{v}',mode]-a['G6',f'{m}_{r}_K{v}',mode] for v in [0,k]}
    for v in [0,k]:out.append(dict(model=m,contrast=f'{l}_minus_{r}_K{v}',readout=mode,**interval(diff[v])))
    out.append(dict(model=m,contrast=f'{l}_minus_{r}_high_minus_zero',readout=mode,**interval(diff[k]-diff[0])))
 return out

def compare(reference,actual,keys):
 expected={tuple(r[k] for k in keys):r for r in reference};computed={tuple(r[k] for k in keys):r for r in actual}
 if set(expected)!=set(computed):raise ValueError('Contrast inventory mismatch')
 for key,r in expected.items():
  for field in ['estimate','ci95_low','ci95_high']:
   if abs(float(r[field])-computed[key][field])>1e-10:raise ValueError(f'Numerical mismatch {key}: {field}')
 return len(expected)

def tex_table(caption,label,headers,rs,align=None):
 align=align or ('l'+'r'*(len(headers)-1))
 return '\n'.join([r'\begin{table}[htbp]\centering\color{blue}\footnotesize',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+align+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule',*[' & '.join(map(str,r))+r' \\' for r in rs],r'\bottomrule\end{tabular}\end{table}'])+'\n'

def render(out,a,regional):
 os.environ.setdefault('MPLCONFIGDIR',str(out/'.mplconfig'))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 def value(m,b,k,mode='generated'):return float(a['G6',f'{m}_{b}_K{k}',mode].mean())*100
 main=[];detail=[]
 for m,label,k in MODELS:
  for b in ['parent','early','middle','late']:
   main.append([label+' '+b,f'{value(m,b,0):.2f}',f'{value(m,b,k):.2f}',f'{value(m,b,0,"valid"):.2f}',f'{value(m,b,k,"valid"):.2f}'])
   arr=a['G6',f'{m}_{b}_K0','generated']
   for seed in range(arr.shape[0]):
    detail.append([label,b,str(seed),*[f'{100*a["G6",f"{m}_{b}_K{v}",r][seed].mean():.2f}' for r in ['generated','valid'] for v in [0,k]]])
 caption=r'\textbf{Regional accuracy and output validity (\%).} Zero-shot and contextual endpoints on AG News: $K=16$ for Llama/DeepSeek, 32 for Mistral and 64 for InternLM. Means weight fits equally. Llama has three fits per band, others one; parents are shared untreated baselines. Invalid outputs count as incorrect. Three paired endpoint draws; zero-shot once per query/adapter. Query counts are 200 for Llama/Mistral and 100 for DeepSeek/InternLM. Whole-prompt supervision is used throughout; this is distinct from the query-label objective in Table~\ref{tab:controlled}.'
 (out/'regional_main_table.tex').write_text(tex_table(caption,'tab:recovery',['Model / band',r'$a(0)$',r'$a(K)$',r'Valid $0$',r'Valid $K$'],main))
 shared=[]
 for m,label in [('base','Base'),('instruct','Instruct'),('hermes','Hermes'),('tulu_sft',r'T\"ulu SFT'),('tulu_dpo',r'T\"ulu DPO'),('tulu_final',r'T\"ulu final')]:
  for k in [0,4,16,32]:shared.append([label,str(k),*[f'{100*a["G12",f"{m}_K{k}",r].mean():.2f}' for r in ['raw','calibrated','generated','valid']]])
 supp=tex_table(r'Common-prompt accuracy and validity (\%). Each checkpoint has 200 queries, three nonzero-$K$ draws, and zero-shot once.','stab:shared_readouts',['Checkpoint','$K$','Raw','Calibrated','Generated','Valid'],shared,'lrrrrr')
 # Split detailed table to keep all 22 fit/parent rows on a page.
 supp+=tex_table(r'Every regional fit and parent (\%). Columns give generated accuracy and validity at zero-shot and the model endpoint. Seed indexes jointly vary calibration and initialization.','stab:regional_detail',['Model','Band','Seed','$a(0)$','$a(K)$','Valid 0','Valid K'],detail,'lllrrrr')
 pair=[]
 for x in regional:
  if x['readout']!='generated' or 'high_minus' in x['contrast']:continue
  pair.append([x['model'].replace('_',r'\_'),x['contrast'].replace('_',r'\_'),f'{100*x["estimate"]:+.2f}',f'[{100*x["ci95_low"]:.2f}, {100*x["ci95_high"]:.2f}]'])
 supp+=tex_table(r'All direct paired regional generated-accuracy endpoint contrasts (percentage points). Pointwise 95\% intervals; other checkpoint intervals condition on one fit. All validity and gap contrasts accompany the data.','stab:regional_pairs',['Model','Contrast','Difference',r'95\% interval'],pair,'llrr')
 (out/'followup_tables.tex').write_text(supp)
 colors=['#666666','#1765a3','#d47510','#34834b'];fig,axes=plt.subplots(1,4,figsize=(13,3.8),sharey=True)
 for ax,(m,label,k) in zip(axes,MODELS):
  for b,c in zip(['parent','early','middle','late'],colors):
   stats=[interval(a['G6',f'{m}_{b}_K{v}','generated']) for v in [0,k]]
   y=np.array([x['estimate']*100 for x in stats]);err=np.array([[y[i]-x['ci95_low']*100 for i,x in enumerate(stats)],[x['ci95_high']*100-y[i] for i,x in enumerate(stats)]])
   ax.errorbar([0,1],y,yerr=np.maximum(err,0),marker='o',color=c,label=b.title(),capsize=3,lw=1.4)
  ax.set_title(label+(' (3 fits/band)' if m=='llama31' else ' (1 fit/band)'),fontsize=10);ax.set_xticks([0,1],['K=0',f'K={k}']);ax.set_ylim(-3,103);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
 axes[0].set_ylabel('Strict generated-label accuracy (%)');handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False);fig.tight_layout(rect=[0,.10,1,1])
 fig.savefig(out/'regional_tradeoffs.pdf',metadata={'CreationDate':None,'ModDate':None});fig.savefig(out/'regional_tradeoffs.png',dpi=170);plt.close(fig)

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=Path('publication/generated'));p.add_argument('--data',type=Path,default=DATA);args=p.parse_args(argv)
 args.out.mkdir(parents=True,exist_ok=True);manifest=verify_inputs(args.data)
 a,n,conf=load_arrays(args.data)
 if n!=manifest['records']:raise ValueError('Record total mismatch')
 print(f'Parsed and paired {n} measurements.',flush=True)
 paired=remaining_contrasts(a,args.data);checkpoint=checkpoint_contrasts(a);regional=regional_contrasts(a)
 checks={}
 for name,computed,keys in [('paired',paired,['family','condition','readout','contrast']),('checkpoint',checkpoint,['model','readout','contrast']),('regional',regional,['model','contrast','readout'])]:
  checks[name]=compare(rows(args.data/f'{name}_reference.csv'),computed,keys);write_rows(args.out/f'{name}_contrasts.csv',computed)
 write_rows(args.out/'confusion_counts.csv',[dict(family=f,condition=c,readout=r,gold=g,prediction=p,count=n) for (f,c,r,g,p),n in sorted(conf.items())])
 render(args.out,a,regional)
 result=dict(status='passed',records=n,independently_recomputed_contrasts=checks,unavailable=manifest['unavailable'],input_manifest_sha256=hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest(),scope='Reanalysis of saved measurements; no new model inference or training. Descriptive companion studies retain their separate source protocols.')
 (args.out/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
 return 0
if __name__=='__main__':raise SystemExit(main())
