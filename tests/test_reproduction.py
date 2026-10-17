"""CPU-only checks for failure reporting, provenance, and run isolation."""
import importlib.util
import json
import os
import re
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded);return loaded
runner=module('reproduction_runner','reproduce.py')
assets=module('reproduction_assets','fetch_reproduction_assets.py')
summary=module('reproduction_summary','summarize_reproduction.py')

class ReproductionTests(unittest.TestCase):
    def test_suite_dependencies(self):
        self.assertEqual([x[0] for x in runner.plan('all')],['rms2d','rms3d','rms3d_extended','fusion3d','fusion3d_extended'])
        self.assertEqual(runner.plan('rms3d')[1][2],'SONGSHAN3_RMS3D_SOURCE')
        self.assertEqual(runner.plan('fusion3d')[1][2],'SONGSHAN3_FUSION_SOURCE')

    def stage(self,code,expect=True,preexisting=False,outside=False):
        with tempfile.TemporaryDirectory() as value:
            root=Path(value);(root/'results').mkdir();run=root/'results/run';run.mkdir()
            result=root/('elsewhere' if outside else 'results/experiment')
            if preexisting:result.mkdir();(result/'exit-code.txt').write_text('0\n')
            record={'stages':[]}
            source=re.sub(r'\bRESULT\b',lambda _:repr(str(result)),code)
            status,path=runner.execute_stage(root,run,record,'test',[sys.executable,'-c',source],os.environ,expect)
            saved=json.loads((run/'run.json').read_text())
            return status,path,saved['stages'][0]

    def test_successful_new_output(self):
        status,path,record=self.stage("import os,pathlib; p=pathlib.Path(RESULT); p.mkdir(); (p/'exit-code.txt').write_text('0'); pathlib.Path(os.environ['SONGSHAN3_STAGE_RESULT_FILE']).write_text(str(p))")
        self.assertEqual(status,0);self.assertEqual(record['status'],'passed');self.assertTrue(path)

    def test_zero_exit_without_output_fails(self):
        status,_,record=self.stage('pass')
        self.assertNotEqual(status,0);self.assertEqual(record['status'],'failed')

    def test_child_error_is_preserved(self):
        status,_,record=self.stage('raise SystemExit(9)')
        self.assertEqual(status,9);self.assertEqual(record['exit_code'],9)

    def test_preexisting_experiment_is_rejected(self):
        status,_,_=self.stage("import os,pathlib; pathlib.Path(os.environ['SONGSHAN3_STAGE_RESULT_FILE']).write_text(RESULT)",preexisting=True)
        self.assertNotEqual(status,0)

    def test_external_experiment_is_rejected(self):
        status,_,_=self.stage("import os,pathlib; p=pathlib.Path(RESULT); p.mkdir(); (p/'exit-code.txt').write_text('0'); pathlib.Path(os.environ['SONGSHAN3_STAGE_RESULT_FILE']).write_text(str(p))",outside=True)
        self.assertNotEqual(status,0)

    def test_failed_experiment_marker_is_rejected(self):
        status,_,_=self.stage("import os,pathlib; p=pathlib.Path(RESULT); p.mkdir(); (p/'exit-code.txt').write_text('3'); pathlib.Path(os.environ['SONGSHAN3_STAGE_RESULT_FILE']).write_text(str(p))")
        self.assertNotEqual(status,0)

    def test_offline_missing_inputs_fail(self):
        with tempfile.TemporaryDirectory() as value:
            root=Path(value)
            with self.assertRaises(RuntimeError):assets.download(root/'missing',{'bytes':1,'sha256':'0'*64},'https://example.invalid',True)
            with self.assertRaises(RuntimeError):assets.checked_path(root,'../escape')

    def test_corrupt_existing_dependency_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as value:
            path=Path(value)/'asset';path.write_bytes(b'corrupt')
            with self.assertRaises(RuntimeError):assets.download(path,{'bytes':7,'sha256':'0'*64},'https://example.invalid',False)
            self.assertEqual(path.read_bytes(),b'corrupt')

    def test_summary_rejects_fabricated_median(self):
        row={'dtype':'float16','shape':[1,128],'repeats':5,'iterations':100,
             'baseline_host_ms':[1]*5,'optimized_host_ms':[.5]*5,
             'baseline_host_median_ms':9,'optimized_host_median_ms':.5,'host_speedup':2}
        with self.assertRaises(RuntimeError):summary.timing_rows('rms2d','main',[row],[('host','api')],Path('new-run'))

if __name__=='__main__':unittest.main()
