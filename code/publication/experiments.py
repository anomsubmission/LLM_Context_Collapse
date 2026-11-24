"""Inspect, prepare and execute the published frozen scientific runtime.

Preparation/validation are CPU-only. Only the explicit run subcommand loads models.
Fresh results live outside the publication checkout and never overwrite evidence.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT/'runtime'

def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def content_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def verify_release(runtime=RUNTIME):
    manifest=read(runtime/'manifest.json')
    for name,digest in manifest['sources'].items():
        if sha(runtime/'sources'/name)!=digest:raise ValueError('Runtime source hash mismatch: '+name)
    if sha(runtime/'registrations.tar.gz')!=manifest['registration_archive_sha256']:
        raise ValueError('Registration archive changed')
    with tarfile.open(runtime/'registrations.tar.gz','r:gz') as archive:
        observed=set()
        for item in archive:
            name=PurePosixPath(item.name)
            if not item.isfile() or name.is_absolute() or '..' in name.parts or item.name in observed:
                raise ValueError('Unsafe or duplicate registration archive entry')
            observed.add(item.name)
            if hashlib.sha256(archive.extractfile(item).read()).hexdigest()!=manifest['registrations'].get(item.name):
                raise ValueError('Registered input hash mismatch: '+item.name)
        if observed!=set(manifest['registrations']):raise ValueError('Incomplete registration archive')
    return manifest

def execution_manifests(manifest):
    """New manifests bind the shipped subset; original freezes remain unchanged."""
    result={}
    common={**manifest['sources'],**{p:h for p,h in manifest['registrations'].items() if p.endswith('/inputs.json')}}
    configs={}
    for slug,job in manifest['jobs'].items():
        reg='revision/'+job['campaign']+'/registration/'
        if job['campaign'].startswith('followup'):
            configs[slug+'.json']=manifest['registrations'][reg+slug+'.json']
        else:
            result[reg+slug+'.freeze.json']=dict(config_content_sha256=content_hash(job['config']),
                files={**common,**manifest['registrations']},
                scope='Fresh replication with original scientific code and inputs; shipped dependency subset.',
                outcome_selection=False,technical_failure_is_not_scientific_null=True)
    result['revision/followup_20261116/registration/runtime_freeze.json']=dict(
        runtime_sources=manifest['sources'],configs=configs,
        status='published_runtime_replication',final_launch_requires_pilot_acceptance=True)
    return result

def prepare(destination,runtime=RUNTIME):
    manifest=verify_release(runtime);destination=Path(destination).resolve()
    if destination==ROOT or ROOT in destination.parents:
        raise ValueError('Use a separate workspace outside the publication checkout')
    if destination.exists():raise FileExistsError('Preparation requires a new directory')
    destination.mkdir(parents=True)
    for name in manifest['sources']:
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(runtime/'sources'/name,target)
    with tarfile.open(runtime/'registrations.tar.gz','r:gz') as archive:
        for item in archive:
            target=destination/item.name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(archive.extractfile(item).read())
    for name,value in execution_manifests(manifest).items():write(destination/name,value)
    write(destination/'replication_manifest.json',dict(
        runtime_manifest_sha256=sha(runtime/'manifest.json'),
        prepared_at=datetime.now(timezone.utc).isoformat(),
        scope='New execution of the registered protocol, not new independent held-out evidence.',
        execution_adjustments=manifest['execution_adjustments']))
    validate_workspace(destination,runtime)
    return destination

def validate_workspace(workspace,runtime=RUNTIME):
    manifest=verify_release(runtime);workspace=Path(workspace).resolve()
    if read(workspace/'replication_manifest.json')['runtime_manifest_sha256']!=sha(runtime/'manifest.json'):
        raise ValueError('Workspace belongs to another runtime release')
    for name,digest in {**manifest['sources'],**manifest['registrations']}.items():
        if sha(workspace/name)!=digest:raise ValueError('Workspace input/source changed: '+name)
    for name,value in execution_manifests(manifest).items():
        if read(workspace/name)!=value:raise ValueError('Execution manifest changed: '+name)
    return manifest

def prerequisites(workspace,slug,manifest):
    job=manifest['jobs'][slug]
    for dep in job['requires']:
        base=workspace/'revision'/job['campaign']/'outputs'/dep
        if not (base/'status.json').is_file() or read(base/'status.json').get('status')!='complete':
            raise ValueError('Run the technical pilot first: '+dep)
        summary=read(base/'summary.json')
        if job['campaign'].startswith('remaining') and not summary.get('technical_checks_passed'):
            raise ValueError('Technical pilot did not pass: '+dep)
    return job

def activate(workspace):
    for rel in ['code','revision/four_day_revision/runtime','revision/followup_20261116/runtime','revision/remaining_20261116/runtime']:
        sys.path.insert(0,str(workspace/rel))
    focused=importlib.import_module('focused_scoring')
    # Portability adjustment only: scientific functions and their bytes are intact.
    focused.CANONICAL_LOCK=workspace/'accelerator.lock'
    return focused

def execute(workspace,slug):
    workspace=Path(workspace).resolve();manifest=validate_workspace(workspace)
    job=prerequisites(workspace,slug,manifest);focused=activate(workspace)
    focused.inherited_accelerator()
    cfg=job['config']
    if job['campaign'].startswith('followup'):
        common=importlib.import_module('common_readout');common.run(slug)
        if cfg['role']=='pilot':
            final=slug.removesuffix('_pilot')+'_final';base=workspace/'revision'/job['campaign']
            # The collector verifies complete finite outputs and no overflow.
            # Acceptance is technical, never based on accuracy or effect direction.
            summary=read(base/'outputs'/slug/'summary.json')
            expected=cfg['n_queries']*(1+(len(cfg['ks'])-1)*len(cfg['draws']))
            if summary['n_records']!=expected or summary['role']!='pilot':raise ValueError('Incomplete technical pilot')
            write(base/'registration'/f'{final.removesuffix("_final")}_pilot_acceptance.json',dict(
                accepted=True,final_config_sha256=sha(base/'registration'/f'{final}.json'),
                pilot_summary_sha256=sha(base/'outputs'/slug/'summary.json'),
                criterion='Complete registered finite/untruncated pilot; no accuracy threshold.',
                provenance='New replication pilot, not the published historical pilot.'))
    else:
        module=importlib.import_module(Path(cfg['runner']).stem)
        cls=module.RegionalJob if cfg['family']=='G6' else module.Job
        task=cls(slug)
        try:module.main(task)
        except BaseException as error:task.fail(error);raise

def run(workspace,slug,dry_run=False):
    workspace=Path(workspace).resolve();manifest=validate_workspace(workspace)
    if slug not in manifest['jobs']:raise ValueError('Unknown published job: '+slug)
    job=manifest['jobs'][slug]
    if dry_run:
        return dict(job=slug,role=job['config']['role'],requires=job['requires'],
            model=job['config']['model'],workspace=str(workspace),models_loaded=False)
    prerequisites(workspace,slug,manifest)
    lock=workspace/'accelerator.lock'
    with lock.open('a+b') as stream:
        try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('Another job owns this workspace accelerator lock') from error
        fd=stream.fileno();os.set_inheritable(fd,True)
        env=dict(os.environ,LLM_SHARED_ACCELERATOR_LOCK_FD=str(fd),PYTHONDONTWRITEBYTECODE='1')
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'_execute',str(workspace),slug],
            env=env,pass_fds=(fd,),check=True,cwd=workspace)
    return dict(job=slug,status='complete',workspace=str(workspace))

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('verify');sub.add_parser('list')
    for name in ['prepare','validate','smoke-imports']:
        sub.add_parser(name).add_argument('workspace',type=Path)
    p=sub.add_parser('run');p.add_argument('workspace',type=Path);p.add_argument('job');p.add_argument('--dry-run',action='store_true')
    p=sub.add_parser('_execute');p.add_argument('workspace',type=Path);p.add_argument('job')
    args=parser.parse_args(argv)
    if args.command=='verify':
        m=verify_release();result=dict(status='passed',source_files=len(m['sources']),jobs=len(m['jobs']))
    elif args.command=='list':
        m=verify_release();result=[dict(job=k,role=v['config']['role'],requires=v['requires']) for k,v in m['jobs'].items()]
    elif args.command=='prepare':result=dict(workspace=str(prepare(args.workspace)),models_loaded=False)
    elif args.command=='validate':validate_workspace(args.workspace);result=dict(status='passed',models_loaded=False)
    elif args.command=='smoke-imports':
        validate_workspace(args.workspace);activate(args.workspace.resolve())
        modules=['common_readout','shared','regional','patching','splice','length',
                 'cliff.backends.hf_local','cliff.experiments.intervention','cliff.experiments.revision_adapter_eval']
        for name in modules:importlib.import_module(name)
        result=dict(status='passed',modules=modules,models_loaded=False)
    elif args.command=='run':result=run(args.workspace,args.job,args.dry_run)
    else:execute(args.workspace,args.job);result=dict(status='complete')
    print(json.dumps(result,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
