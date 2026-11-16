"""Create an unchecked library with 40581f7, then check/settle/evict with current code.

PYTHONPATH=src python docs/validation/run_review_nine_upgrade.py --legacy-src /path/to/40581f7/src
The supplied legacy checkout is read only; all executions use a temporary home.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from or_harness.api import ORHarness
from or_harness.strategy.embedding_index import LocalHashEmbeddingBackend
from or_harness.strategy.evidence_access import resolve_execution
from or_harness.world_model.conditions import get_value
from or_harness.world_model.hplus import get_job,receipt_observation
from or_harness.world_model.hplus_support import _report_at
from or_harness.world_model.episode_closeout import enforce_evidence_window,CalibrationPolicy,EvidenceWindowPolicy
from or_harness.world_model.induction_gate import meta_read


SEED=r'''
import json,sys
from pathlib import Path
from or_harness.api import ORHarness
from or_harness.strategy.embedding_index import LocalHashEmbeddingBackend
from run_review_eight_demo import SubstituteProvider,problem,METHOD
home=sys.argv[1]
h=ORHarness(home=home,world_model=SubstituteProvider(),embedding=LocalHashEmbeddingBackend())
h.world_model.decision=1
t=problem('upgrade-40581',641,130)
p=h.predict_strategy_outcome(t,{'strategy_id':'endpoints','solver':'python','method':METHOD},t['task_id'])
work=Path(home)/'work';work.mkdir()
script=work/'solve.py'
script.write_text("import json\njson.dump({'status':'optimal','objective_value':90,'variables':{'X':470,'Y':170}},open('result.json','w'))\n")
r=h.execute(t,None,str(script),str(work),episode_id=t['task_id'],prediction_id=p.prediction_id)
h.record(r)
closed=h.close_episode(t['task_id'],t['task_id'])
print(json.dumps({'execution_id':r.execution_id,'induction_id':closed['induction_schedule']['pending_ids'][0]}))
h.close()
'''


def run(legacy_src):
    legacy_src=Path(legacy_src).resolve()
    if not (legacy_src/'or_harness'/'api.py').is_file():raise ValueError('--legacy-src must contain or_harness/api.py')
    with tempfile.TemporaryDirectory(prefix='or-review-nine-upgrade-') as home:
        env={**os.environ,'PYTHONPATH':str(legacy_src)+os.pathsep+str(Path(__file__).resolve().parent),
             'PYTHONDONTWRITEBYTECODE':'1'}
        seed=subprocess.run([sys.executable,'-c',SEED,home],env=env,check=True,capture_output=True,text=True)
        ids=json.loads(seed.stdout)
        eid,j=ids['execution_id'],ids['induction_id']
        h=ORHarness(home=home,embedding=LocalHashEmbeddingBackend())
        try:
            old_ref=meta_read(h.store,'rule_evidence_latest:'+eid)
            assert old_ref and not get_value(h.store,old_ref)['record']['execution_features'].get('task_check')
            before=time.time()
            checked=h.check_task_result(eid,{'reference_objective':90,'reference_source':'independent'})
            h.induction_material(induction_id=j,related_top_k=0)
            result=h.induce(induction_id=j,notes=['Agent reviewed the checked workflow; no reusable increment.'],review={
                'outcome':'no_increment','comparison':{'sufficient':True,'substantive':False,
                    'source':'demo reviewer compared the actual checked result and frozen input',
                    'assessment':'No reusable increment within the reviewed scope.',
                    'evidence_refs':[get_job(h,j)['source_fact_ref']]}})
            assert receipt_observation(h,get_job(h,j))['label']==0
            enforce_evidence_window(h,policy=EvidenceWindowPolicy(window_episodes=0),
                calibration_policy=CalibrationPolicy(window=0,late_check_grace_days=0))
            assert h.bank.get(eid) is None
        finally:h.close()
        h=ORHarness(home=home,embedding=LocalHashEmbeddingBackend())
        try:
            record=resolve_execution(h,eid)
            report=record.execution_features['task_check']
            label=receipt_observation(h,get_job(h,j))['label']
            assert report['state']=='passed' and report['action_id']==checked['action_id'] and label==0
            assert _report_at(h,record,before)=={}
            return {'legacy_version':'40581f7','execution_id':eid,'induction_id':j,'legacy_full_ref':old_ref,
                'current_fact_ref':meta_read(h.store,'rule_fact_latest:'+eid),'checker_action_id':report['action_id'],
                'live_execution_evicted':h.bank.get(eid) is None,'checker_after_restart':report['state'],
                'label_after_restart':label,'historical_checker_before_correction':_report_at(h,record,before),
                'receipt_ref':result['induction_receipt']['receipt_ref'],
                'provider':'substitute; this verifies upgrade and evidence plumbing only'}
        finally:h.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-src',required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.legacy_src),ensure_ascii=False,indent=2))
