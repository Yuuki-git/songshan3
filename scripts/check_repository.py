"""Validate the public checkout without importing GPU packages or downloading."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

def normalized(name):return re.sub('[-_.]+','-',name).lower()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bash',default=os.environ.get('BASH_BIN','bash'))
    p.add_argument('--allow-untracked',action='store_true')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    files=[s.strip() for s in (root/'manifests/reproduction-files.txt').read_text().splitlines() if s.strip() and not s.startswith('#')]
    assert len(files)==len(set(files)),'Duplicate release path'
    for name in files:
        path=(root/name).resolve();assert path.is_relative_to(root) and path.is_file(),name
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8-sig'),filename=name)
        if path.suffix=='.sh':
            assert b'\r' not in path.read_bytes(),f'Non-LF shell script: {name}'
            subprocess.run([args.bash,'-n',path.as_posix()],check=True)
    upstream=json.loads((root/'manifests/upstreams.json').read_text())
    assert len(upstream['repositories'])==4 and len(upstream['archives'])==3
    assert all(re.fullmatch('[0-9a-f]{40}',r['commit']) and r['url'].startswith('https://github.com/') for r in upstream['repositories'])
    wheels=json.loads((root/'manifests/workstation-wheels.json').read_text())['wheels']
    expected={normalized(w['filename'].split('-')[0]):(w['filename'].split('-')[1],w['sha256']) for w in wheels}
    assert len(expected)==len(wheels)==17
    simple={normalized(s.split('==')[0]):s.split('==')[1] for s in (root/'requirements.txt').read_text().splitlines() if s and not s.startswith('#')}
    locked={}
    for line in (root/'requirements-lock.txt').read_text().splitlines():
        if not line or line.startswith('#'):continue
        m=re.fullmatch(r'([^= ]+)==([^ ]+) --hash=sha256:([0-9a-f]{64})',line);assert m,line
        locked[normalized(m[1])]=(m[2],m[3])
    assert locked==expected and simple=={k:v[0] for k,v in expected.items()},'Dependency manifests disagree'
    for relative in ('run_hygon_rmsnorm3d_extended.sh','run_hygon_fusion3d_extended.sh'):
        content=(root/'scripts'/relative).read_text()
        assert not re.search(r'results/[^\s\"]*-20\d{6}T',content),'Dated fallback in public launcher'
    assert all(name not in files for name in ('scripts/remote.py','scripts/push_task_files.py','docs/WORKLOG.md','AGENTS.md'))
    if not args.allow_untracked:
        tracked=set(subprocess.check_output(['git','-C',str(root),'ls-files'],text=True).splitlines())
        assert tracked==set(files),f'Release inventory differs: extra={tracked-set(files)}, missing={set(files)-tracked}'
    print(json.dumps({'passed':True,'release_files':len(files),'pinned_wheels':len(wheels),'gpu_executed':False}))

if __name__=='__main__':main()
