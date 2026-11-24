"""Integrity guards for measurement identity, output validity and contrast retention."""
import gzip,json,hashlib,tempfile,unittest
from pathlib import Path
import numpy as np
from .reproduce import verify_inputs,verify_regional_audits,load_arrays,interval,compare,remaining_contrasts,DATA
class IntegrityTests(unittest.TestCase):
 def test_changed_measurements_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);f=p/'records.json';f.write_text('original')
   (p/'manifest.json').write_text(json.dumps({'files':{'records.json':hashlib.sha256(f.read_bytes()).hexdigest()}}));f.write_text('changed')
   with self.assertRaisesRegex(ValueError,'hash mismatch'):verify_inputs(p)
 def test_zero_shot_unit_and_invalid_continuation(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'query_orders.json').write_text(json.dumps({'remaining_20261116':['q1','q2'],'followup_20261116':[]}))
   with gzip.open(p/'measurements.jsonl.gz','wt') as f:
    for q,text in [('q1','Sports'),('q2','Sports\nText: another example')]:f.write(json.dumps(dict(family='G6',condition='m_early_K0',seed=0,draw=0,query=q,gold='Sports',text=text))+'\n')
   a,n,_=load_arrays(p);self.assertEqual(n,2);self.assertEqual(a['G6','m_early_K0','generated'].shape,(1,1,2));self.assertEqual(a['G6','m_early_K0','valid'].mean(),.5)
 def test_changed_or_omitted_adverse_result_rejected(self):
  r=[dict(condition='adverse',estimate=-.7,ci95_low=-.8,ci95_high=-.6)]
  with self.assertRaisesRegex(ValueError,'inventory'):compare(r,[],['condition'])
  with self.assertRaisesRegex(ValueError,'Numerical'):compare(r,[dict(r[0],estimate=.7)],['condition'])
 def test_unknown_query_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'query_orders.json').write_text(json.dumps({'remaining_20261116':['q1'],'followup_20261116':[]}))
   with gzip.open(p/'measurements.jsonl.gz','wt') as f:f.write(json.dumps(dict(family='G6',query='unregistered'))+'\n')
   with self.assertRaisesRegex(ValueError,'Unknown query'):load_arrays(p)
 def test_k0_not_spurious_precision(self):
  a=np.array([[[0.,1.,0.,1.,1.]]]);x=interval(a);y=interval(np.repeat(a,3,axis=1))
  self.assertEqual(x['n_draws'],1);self.assertGreater(x['ci95_high']-x['ci95_low'],.3);self.assertGreater(y['ci95_high']-y['ci95_low'],.3)
 def test_exported_zero_shot_count_is_observed_count(self):
  # A one-draw baseline is shared across three paired endpoint draws. Keep
  # the registered interval computation but do not report three observations.
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'language.json').write_text(json.dumps({
    f'G5_{recipient}_{band}_final':[dict(span_id='span',input_token_ids_hash='hash',nll_per_token=1.)] for recipient in ['instruct','hermes']
    for band in ['parent','early','middle','late']}))
   key=('G6','example_parent_K0','generated');a=np.array([[[0.,1.,0.,1.,1.]]])
   rows=remaining_contrasts({key:a},p)
   row=next(r for r in rows if r['condition']==key[1])
   self.assertEqual(row['n_draws'],1)
   old=interval(np.repeat(a,3,axis=1))
   for field in ['estimate','ci95_low','ci95_high']:self.assertEqual(row[field],old[field])
 def test_calibration_count_must_match_registered_rows(self):
  import shutil
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   for name in ['adapter_audit.json','regional_training_audits.json.gz','protocol_companions.json.gz']:shutil.copyfile(DATA/name,p/name)
   verify_regional_audits(p)
   rows=json.loads((p/'adapter_audit.json').read_text());rows[0]['calibration_rows']=1
   (p/'adapter_audit.json').write_text(json.dumps(rows))
   with self.assertRaisesRegex(ValueError,'Calibration row count'):verify_regional_audits(p)
if __name__=='__main__':unittest.main()
