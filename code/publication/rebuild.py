"""Rebuild current figures/tables and check every TeX dependency.

Run from code: python -m publication.rebuild [--compile]
Raw logit-lens curves survive only as vector artwork; these are preserved, not
represented as recomputed per-query measurements. No fresh model runs occur.
"""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess,sys,shutil
from . import reproduce
ROOT=Path(__file__).resolve().parents[2]
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--compile',action='store_true');p.add_argument('--skip-followup',action='store_true');args=p.parse_args(argv)
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MPLCONFIGDIR=str(ROOT/'code/publication/generated/.mplconfig'),XDG_CACHE_HOME=str(ROOT/'code/.cache'))
 out=ROOT/'code/publication/generated';out.mkdir(parents=True,exist_ok=True)
 documents=['article_cac.tex','supplementary_ncs.tex']
 if (ROOT/'Response.tex').exists():documents.append('Response.tex')
 if not args.skip_followup:reproduce.main(['--out',str(out)])
 else:
  validation=json.loads((out/'validation.json').read_text())
  if validation['status']!='passed' or validation['input_manifest_sha256']!=hashlib.sha256((reproduce.DATA/'manifest.json').read_bytes()).hexdigest():raise ValueError('Follow-up not validated for current inputs')
  reproduce.verify_inputs(reproduce.DATA)
 commands=[([sys.executable,str(ROOT/x)],ROOT) for x in ['revision/final_integration/generate_final_behavioral_panel.py','revision/final_integration/generate_final_supplement.py']]
 commands += [([sys.executable,'-m','cliff.revision_mechanism_figure'],ROOT/'code'),([sys.executable,str(ROOT/'revision/final_integration/finalize_mechanistic_legend.py')],ROOT)]
 for command,cwd in commands:subprocess.run(command,cwd=cwd,env=env,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 paths=set();labels={};refs={}
 def visit(path,seen):
  if path in seen:return ''
  seen.add(path);paths.add(path);text=re.sub(r'(?m)(?<!\\)%.*','',path.read_text())
  for name in re.findall(r'\\input\{([^}]+)\}',text):text+='\n'+visit(ROOT/name,seen)
  for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',text):
   f=ROOT/name
   if not f.exists():raise FileNotFoundError(f)
   paths.add(f)
  return text
 for name in documents:
  text=visit(ROOT/name,set());ls=re.findall(r'\\label\{([^}]+)\}',text);rs=re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',text)
  if len(ls)!=len(set(ls)) or set(rs)-set(ls):raise ValueError('Unresolved/duplicate labels in '+name)
  labels[name]=len(ls);refs[name]=len(rs)
 if args.compile:
  compiler=shutil.which('tectonic') or '/opt/homebrew/bin/tectonic';build=ROOT/'revision/manuscript_20261120/build';build.mkdir(parents=True,exist_ok=True)
  for name in documents:
   with (out/(Path(name).stem+'_compile.log')).open('w') as log:subprocess.run([compiler,'-X','compile','--keep-logs','--outdir',str(build),name],cwd=ROOT,env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
 report=dict(status='passed',labels=labels,references=refs,document_dependencies={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(paths)},scope='Current follow-up analyses, behavioral panel, objective/readout supplementary tables, mechanistic state arrays. Raw logit-lens and ancillary supplementary artwork are retained because per-query sources are unavailable; no fresh training is claimed.')
 (out/'document_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='document_dependencies'}));return 0
if __name__=='__main__':raise SystemExit(main())
