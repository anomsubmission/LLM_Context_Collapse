"""CPU-only release tests: isolation, missing prerequisites and source integrity."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from . import experiments as ex

class RuntimeTests(unittest.TestCase):
    def test_packaged_workspace_is_complete_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            workspace=ex.prepare(Path(d)/'run')
            manifest=ex.validate_workspace(workspace)
            self.assertEqual(len(manifest['jobs']),58)
            self.assertEqual(sum(j['config']['role']=='final' for j in manifest['jobs'].values()),38)
            for slug in manifest['jobs']:
                result=ex.run(workspace,slug,dry_run=True)
                self.assertFalse(result['models_loaded'])
            with self.assertRaises(FileExistsError):ex.prepare(workspace)
            with self.assertRaisesRegex(ValueError,'technical pilot first'):
                ex.run(workspace,'instruct_final')
            altered=workspace/'code/cliff/prompting.py'
            altered.write_text(altered.read_text()+'\n# changed\n')
            with self.assertRaisesRegex(ValueError,'source changed'):ex.validate_workspace(workspace)

    def test_source_imports_and_inherited_lock(self):
        with tempfile.TemporaryDirectory() as d:
            workspace=ex.prepare(Path(d)/'run')
            subprocess.run([sys.executable,'-m','publication.experiments','smoke-imports',str(workspace)],
                           check=True,stdout=subprocess.PIPE,text=True)
            # Exercise the original inherited-descriptor guard in a subprocess,
            # using the portable path adjustment, without loading model weights.
            import fcntl,os
            with (workspace/'accelerator.lock').open('a+b') as lock:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                env=dict(os.environ,LLM_SHARED_ACCELERATOR_LOCK_FD=str(lock.fileno()))
                code='from pathlib import Path; from publication.experiments import activate; activate(Path('+repr(str(workspace))+')).inherited_accelerator()'
                subprocess.run([sys.executable,'-c',code],env=env,pass_fds=(lock.fileno(),),check=True)

if __name__=='__main__':unittest.main()
