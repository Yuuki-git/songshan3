"""Validate newly generated suite outputs and recompute their timing tables."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def require(condition, message):
    if not condition:raise RuntimeError(message)

def environment(directory,root):
    b=load(directory/'environment.before.json');a=load(directory/'environment.after.json')
    checks={key:b[key]==a[key] for key in ('gpu','libraries','packages','global_packages','torch','loaded_infini_libraries','benchmark_sources')}
    checks['base_packages_preserved']=b['global_package_list_unchanged'] is True and a['global_package_list_unchanged'] is True
    checks['independent_venv']=Path(a['executable']).is_relative_to(root/'venvs/h3')
    checks['independent_libraries']=bool(a['loaded_infini_libraries']) and all(Path(p).is_relative_to(root/'infini/lib') for p in a['loaded_infini_libraries'])
    for name in ('InfiniCore','InfiniLM','ninetoothed'):
        checks[name+'_unchanged']=b['sources'][name]==a['sources'][name]
        checks[name+'_clean']=a['sources'][name]['status']['exit_code']==0 and not a['sources'][name]['status']['stdout'].strip()
    require(all(checks.values()),f'Environment invariants failed: {directory}: {checks}')
    return checks

def timing_rows(suite,phase,records,kinds,source):
    output=[]
    for record in records:
        require(record['repeats']==5 and record['iterations']==100,'Unexpected timing protocol')
        for kind,label in kinds:
            medians={}
            for mode in ('baseline','optimized'):
                raw=record[f'{mode}_{kind}_ms']
                require(len(raw)==5 and all(math.isfinite(v) and v>0 for v in raw),'Invalid raw timing')
                median=statistics.median(raw);medians[mode]=median
                require(math.isclose(median,record[f'{mode}_{kind}_median_ms'],rel_tol=1e-12,abs_tol=1e-12),'Median mismatch')
            speedup=medians['baseline']/medians['optimized']
            require(math.isclose(speedup,record[f'{kind}_speedup'],rel_tol=1e-12,abs_tol=1e-12),'Speedup mismatch')
            output.append(dict(suite=suite,phase=phase,timing=label,op=record.get('op','rms_norm'),
                dtype=record['dtype'],weight_dtype=record.get('weight_dtype',''),shape=json.dumps(record['shape']),
                baseline_ms=medians['baseline'],optimized_ms=medians['optimized'],speedup=speedup,source=str(source)))
    return output

def summarize(run_path):
    run=load(run_path);root=Path(run['root']).resolve();output=run_path.parent
    stages={r['name']:r for r in run['stages'] if r['name']!='summary'}
    require(stages and all(r['status']=='passed' for r in stages.values()),'A preceding stage did not pass')
    require(all(sha(root/p)==h for p,h in run['source_sha256'].items()),'Reproduction sources changed during the run')
    summary={'schema':'songshan3-reproduction-summary-v1','suite':run['suite'],'scope':run['scope'],
             'model_end_to_end':False,'checks':{},'environment_invariants':{}}
    times=[];checksums={}
    for name,stage in stages.items():
        if 'result' not in stage:continue
        d=Path(stage['result']).resolve()
        require(d.is_relative_to(root/'results'),'Result outside this checkout')
        require((d/'exit-code.txt').read_text().strip()=='0','Nonzero experiment exit')
        summary['environment_invariants'][name]=environment(d,root)
        if name=='rms2d':
            a=load(d/'acceptance/acceptance.json');c=rows(d/'acceptance/correctness.jsonl')
            require(a['correctness_passed']==a['correctness_total']==len(c)==75,'RMS2D correctness count')
            require(all(r['passed'] for r in c),'RMS2D correctness failure')
            require(a['frozen_rms_passed']==a['frozen_rms_total']==40 and a['frozen_native']==20,'RMS2D frozen cases')
            require(a['restored'] and a['graph']['passed'] and a['graph']['replay_with_changed_input'],'RMS2D replay/restore')
            summary['checks'][name]={'four_path_cases':75,'frozen_cases':40,'native_frozen_cases':20,'core_list_replay':True}
            times+=timing_rows('rms2d','main',a['paired'],[('host','synchronized_api_host'),('event','core_api_stream_event')],d)
        elif name=='rms3d':
            a=load(d/'acceptance/acceptance.json');c=rows(d/'acceptance/correctness.jsonl');p=load(d/'probe/probe.json')
            require(p['passed']==p['total']==40,'RMS3D candidate checks')
            require(a['correctness_passed']==a['correctness_total']==len(c)==40,'RMS3D correctness count')
            require(all(r['passed'] and r['input_unchanged'] and r['output_guards'] and len(r['paths'])==4 for r in c),'RMS3D correctness failure')
            require(a['restored'] and a['fallback_calls']==0 and all(r['passed'] for r in a['inplace']),'RMS3D dispatch/inplace')
            require(len(a['graphs'])==15 and all(r['baseline'] and r['optimized'] and r['changed_input'] for r in a['graphs']),'RMS3D list replay')
            summary['checks'][name]={'candidate_checks':40,'four_path_cases':40,'outputs':160,'core_list_replays':15}
            times+=timing_rows('rms3d','main',a['paired'],[('api','core_api_stream_event'),('graph','core_list_stream_event')],d)
        elif name=='rms3d_extended':
            a=load(d/'extended/extended.json');g=load(d/'hipgraph/graph-benchmark.json');c=load(d/'standalone.log')
            require(a['passed']==a['total']==20 and a['real_original_fallbacks']==15 and a['restored'],'RMS3D extended checks')
            require(c['standalone_rms3d_passed'] and c['output_guards'] and c['restored'] and not c['python_interpreter'],'RMS3D standalone')
            require(g['cases']==len(g['paired'])==15 and g['restored'] and all(r['changed_input_replay'] for r in g['paired']),'RMS3D HIP graph')
            require(Path(g['source']).resolve()==Path(stages['rms3d']['result']).resolve(),'RMS3D reused another run')
            summary['checks'][name]={'extended_checks':20,'real_fallbacks':15,'hipgraph_cases':15,'standalone_cpp':True}
            times+=timing_rows('rms3d','extended',g['paired'],[(k,k) for k in ('api_host','api_event','hipgraph_host','hipgraph_event')],d)
        elif name=='fusion3d':
            a=load(d/'acceptance/acceptance.json');c=rows(d/'acceptance/correctness.jsonl');e=rows(d/'acceptance/extra.jsonl')
            require(a['five_path_cases']==len(c)==28 and a['five_path_outputs']==140,'Fusion five-path count')
            require(all(r['passed'] and r['bitwise'] and r['inputs_unchanged'] and r['output_guards'] and r['atol']==r['rtol']==0 for r in c),'Fusion bitwise correctness')
            require(a['extra_cases']==len(e)==28 and all(r['passed'] for r in e),'Fusion extra cases')
            require(a['registration_cleared'] and a['native_resources_released'] and a['enable_fp_fusion'] is False,'Fusion resources/compiler option')
            summary['checks'][name]={'five_path_cases':28,'bitwise_outputs':140,'extra_checks':28,'enable_fp_fusion':False}
        elif name=='fusion3d_extended':
            a=load(d/'benchmark/benchmark.json');c=load(d/'standalone.log')
            require(c['standalone_fusion_passed'] and c['bitwise'] and c['output_guards'] and not c['python_interpreter'],'Fusion standalone')
            require(c['original_calls_per_subgraph']==10 and c['fused_calls_per_subgraph']==4,'Fusion launch count')
            require(a['cases']==len(a['paired'])==18 and a['cleared'] and a['native_resources_released'],'Fusion benchmark cases')
            require(len(a['extra_checks'])==4 and all(r['passed'] for r in a['extra_checks']),'Fusion overlap guards')
            require(all(r['bitwise'] and r['changed_input_replays'] for r in a['paired']),'Fusion changed-input replay')
            require(Path(a['source']).resolve()==Path(stages['fusion3d']['result']).resolve(),'Fusion reused another run')
            summary['checks'][name]={'paired_cases':18,'alias_guards':4,'standalone_cpp':True,'subgraph_original_calls':10,'subgraph_fused_calls':4}
            times+=timing_rows('fusion3d','extended',a['paired'],[(k,k) for k in ('api_host','api_event','list_host','list_event','hipgraph_host','hipgraph_event')],d)
        for path in sorted(d.rglob('*')):
            if path.is_file():checksums[path.relative_to(root).as_posix()]=sha(path)
    require(summary['checks'],'No operator experiments to summarize')
    summary.update(paired_timing_rows=len(times),verified_files=len(checksums),passed=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'checksums.json').write_text(json.dumps(checksums,indent=2)+'\n')
    with (output/'paired-summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(times[0]));writer.writeheader();writer.writerows(times)
    print(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    summarize(p.parse_args().run.resolve())
