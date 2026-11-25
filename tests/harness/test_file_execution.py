"""File declarations -> selected prediction -> confined execution -> evidence.

All instances are independently constructed. No benchmark answers or real model
risk judgements are used. Optional numerical-reader tests run when installed.
"""
import copy
import importlib.util
import io
import json
import os
import textwrap
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from helpers import HarnessTestCase
from g_fixtures import GProvider, G_PAYLOAD
from or_harness.api import ORHarness
from or_harness.cli import _load_task_arg, main
from or_harness.core.artifacts import read
from or_harness.execution.inputs import prepare_task_inputs
from or_harness.strategy.embedding_index import LocalHashEmbeddingBackend
from or_harness.world_model.hplus import jobs


class FileCase(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.provider = GProvider()
        self.provider.payload = copy.deepcopy(G_PAYLOAD)
        self.provider.payload["capability_gain"] = {"decision": 0, "basis": ["synthetic execution fixture"]}
        self.provider.payload["risk"] = {"events": [
            {"event": name, "probability": .1} for name in
            ("implementation_failure", "timeout", "environment_failure")]}
        self.h = ORHarness(home=Path(self.home) / "memory", world_model=self.provider,
                           embedding=LocalHashEmbeddingBackend())
        self.addCleanup(self.h.close)
        self.work = Path(self.home) / "submission"
        self.work.mkdir()
        self.task = {"task_id": "files", "family": "set_cover",
                     "text": "Read only the declared public inputs and solve the stated task."}

    def resource(self, name, data):
        source = Path(self.home) / "public" / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(data)
        self.task.setdefault("input_resources", []).append({"source": str(source), "path": name})
        return source

    def plan(self, method=None):
        plan = self.h.plan_next(self.task, "ep", candidates=[{
            "solver": "python", "name": "file strategy", "steps": ["read files and compute result"],
            **({"method": method} if method else {})}])
        self.assertIsNotNone(plan["selected_prediction_id"], plan)
        return plan["selected_prediction_id"]

    def execute(self, source, prediction=None):
        script = self.work / "solve.py"
        script.write_text(textwrap.dedent(source))
        return self.h.execute(self.task, code_path=str(script), workspace=str(self.work),
                              prediction_id=prediction or self.plan())

    def assert_objective(self, source, expected):
        record = self.execute(source)
        self.assertTrue(record.quality["feasible"], [f.error for f in record.failures])
        self.assertEqual(record.quality["objective"], expected)
        self.assertTrue(record.execution_features["prediction_binding"]["bound"])
        self.assertEqual(self.h.actions.get(record.action_id).linked_execution_id, record.execution_id)
        artifacts = record.execution_features["artifacts"]
        self.assertIn("submitted_script", artifacts)
        self.assertIn("workspace_result", artifacts)
        self.assertEqual(json.loads(read(self.h.home, artifacts["workspace_result"]))["objective_value"], expected)
        return record


class TestFileInputs(FileCase):
    def test_single_csv(self):
        self.resource("nodes.csv", "cost\n2\n3\n")
        r = self.assert_objective("""
            import csv,json
            with open('inputs/nodes.csv') as f:
                value=sum(int(row['cost']) for row in csv.DictReader(f))
            json.dump({'status':'optimal','objective_value':value},open('result.json','w'))
        """, 5)
        manifest = r.execution_features["input_resources"]
        self.assertEqual(manifest[0]["bytes"], 9)
        self.assertEqual(read(self.h.home, manifest[0]["artifact"]), b"cost\n2\n3\n")
        self.assertEqual(self.h.actions.get(r.action_id).params["input_resources"], manifest)
        self.assertNotIn("cost\\n2", json.dumps(self.provider.requests))

    def test_related_csv_json_dynamic_path_subdirectory_and_txt(self):
        self.resource("network/nodes.csv", "id,cost\na,2\nb,3\n")
        self.resource("network/edges.csv", "from,to\na,b\n")
        self.resource("config.json", '{"multiplier":4}')
        self.resource("instance.dat", "7\n")
        self.assert_objective("""
            from pathlib import Path
            import csv,json
            root=Path('inputs')
            paths={name:root/'network'/name for name in ('nodes.csv','edges.csv')}
            with paths['nodes.csv'].open() as f:
                costs={r['id']:int(r['cost']) for r in csv.DictReader(f)}
            with open(paths['edges.csv']) as f:
                value=sum(costs[e['from']]+costs[e['to']] for e in csv.DictReader(f))
            with open(root/'config.json') as f:
                factor=json.load(f)['multiplier']
            value=value*factor+int((root/'instance.dat').read_text())
            (Path('scratch')/'progress.txt').write_text('read four related inputs')
            (Path('result.json')).write_text(json.dumps({'status':'optimal','objective_value':value}))
        """, 27)

    def test_large_input_not_in_source_or_prediction(self):
        self.resource("large.txt", "1\n" * 600000)
        r = self.assert_objective("""
            import json
            filename='large.txt'
            with open('inputs/'+filename) as f:
                total=sum(int(line) for line in f)
            json.dump({'status':'optimal','objective_value':total},open('result.json','w'))
        """, 600000)
        self.assertLess(len(json.dumps(self.provider.requests)), 80000)
        self.assertLess(len(read(self.h.home, r.execution_features['artifacts']['submitted_script'])), 1000)
        self.assertGreater(r.execution_features['input_resources'][0]['bytes'], 1000000)
        self.h.record(r)
        self.assertEqual(self.h.bank.get(r.execution_id).execution_features['input_resources'],
                         r.execution_features['input_resources'])

    def test_plain_text_compatibility(self):
        self.assert_objective("import json\njson.dump({'status':'optimal','objective_value':3},open('result.json','w'))", 3)

    def test_task_file_relative_sources_and_cli_execute(self):
        self.resource("sub/nodes.csv", "cost\n2\n3\n")
        task_file = Path(self.home) / "public" / "task.json"
        raw = {**self.task, "input_resources": [{"path":"sub/nodes.csv", "source":"sub/nodes.csv"}]}
        task_file.write_text(json.dumps(raw))
        self.task = _load_task_arg(str(task_file))
        prediction = self.plan()
        script = self.work / "solve.py"
        script.write_text("import csv,json\njson.dump({'status':'optimal','objective_value':sum(int(r['cost']) for r in csv.DictReader(open('inputs/sub/nodes.csv')))},open('result.json','w'))")
        with redirect_stdout(io.StringIO()) as output:
            code = main(["--home",str(self.h.home),"execute","--task",str(task_file),
                         "--prediction",prediction,"--code",str(script),"--workspace",str(self.work)])
        body = json.loads(output.getvalue())
        self.assertEqual(code, 0, body)
        self.assertEqual(body["result"]["execution"]["quality"]["objective"], 5)

    def test_invalid_declarations_do_not_authorize_directories_or_traversal(self):
        source = self.resource("ok.txt", "1")
        for name in ("../outside", "/absolute", "a/../b", "a//b", "a/./b", "a\\b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                prepare_task_inputs({**self.task, "input_resources":[{"source":str(source),"path":name}]}, self.h.home)
        with self.assertRaises(ValueError):
            prepare_task_inputs({**self.task, "input_resources":[{"source":str(source.parent),"path":"dir"}]}, self.h.home)
        with self.assertRaises(ValueError):
            prepare_task_inputs({**self.task, "input_resources":[{"source":str(source),"path":"a"},
                {"source":str(source),"path":"a/b"}]}, self.h.home)

    def test_content_artifact_is_deduplicated(self):
        self.resource("first.txt", "same content")
        self.resource("second.txt", "same content")
        prepared = prepare_task_inputs(self.task, self.h.home)
        a,b = prepared['input_resources']
        self.assertEqual(a['artifact']['path'], b['artifact']['path'])
        self.assertNotEqual(a['path'], b['path'])


class TestFileSecurity(FileCase):
    def test_undeclared_host_file_other_task_and_memory_are_inaccessible(self):
        secret = Path(self.home) / "other-task.txt"
        secret.write_text("private input")
        memory = Path(self.h.home) / "bank_texts.json"
        memory.write_text("private knowledge")
        for path in (str(secret), str(memory), "/etc/passwd", "../other-task.txt", "inputs/undeclared.txt"):
            with self.subTest(path=path):
                r = self.execute("from pathlib import Path\npath="+repr(path)+"\nPath(path).read_text()\n")
                self.assertEqual(r.quality['status'], 'error')
                self.assertFalse(r.quality['feasible'])
                self.assertNotIn('private input', json.dumps(r.to_dict()))
        self.assertEqual(secret.read_text(), "private input")

    def test_inputs_cannot_be_written_truncated_or_unlinked(self):
        original = self.resource("data.txt", "immutable")
        source = """
            from pathlib import Path
            import ctypes,json
            p=Path('inputs/data.txt')
            denied=0
            for operation in (lambda:p.write_text('bad'),lambda:p.unlink()):
                try: operation()
                except OSError: denied+=1
            libc=ctypes.CDLL(None,use_errno=True)
            assert libc.truncate(b'inputs/data.txt',0)==-1
            # Native open bypasses Python auditing; read-only mounts still apply.
            assert libc.open(b'inputs/data.txt',1|512)==-1
            assert p.read_text()=='immutable'
            json.dump({'status':'optimal','objective_value':denied},open('result.json','w'))
        """
        self.assert_objective(source, 2)
        self.assertEqual(original.read_text(), "immutable")

    def test_symlink_sources_and_parent_symlinks_are_rejected(self):
        source = self.resource("real.txt", "input")
        link = Path(self.home) / "linked.txt"
        link.symlink_to(source)
        directory_link = Path(self.home) / "linked-dir"
        directory_link.symlink_to(source.parent, target_is_directory=True)
        for target in (link, directory_link / source.name):
            with self.subTest(target=target), self.assertRaises(OSError):
                prepare_task_inputs({**self.task,"input_resources":[{"source":str(target),"path":"file.txt"}]},self.h.home)

    def test_symlink_escape_and_result_collection_do_not_follow_host_links(self):
        secret = Path(self.home) / "secret.json"
        secret.write_text('{"status":"optimal","objective_value":999}')
        source = """
            from pathlib import Path
            import os
            target=SECRET
            getattr(os,'symlink')(target,'result.json')
        """.replace('SECRET',repr(str(secret)))
        r = self.execute(source)
        self.assertEqual(r.quality['status'], 'error')
        self.assertNotIn('workspace_result', r.execution_features['artifacts'])
        self.assertIsNone(r.quality['objective'])
        self.assertEqual(json.loads(secret.read_text())['objective_value'],999)

    def test_native_read_network_fork_and_exec_cannot_bypass_ast(self):
        secret = Path(self.home) / "secret.txt"
        secret.write_text('secret')
        source = """
            import ctypes,json
            libc=ctypes.CDLL(None,use_errno=True)
            assert libc.open(SECRET,0)==-1
            assert libc.open(b'/etc/passwd',0)==-1
            assert libc.open(b'/work/undeclared-output',1|64,384)==-1
            assert libc.open(b'/outside-output',1|64,384)==-1
            assert libc.socket(2,1,0)==-1
            assert libc.fork()==-1
            assert libc.execve(b'/bin/sh',None,None)==-1
            json.dump({'status':'optimal','objective_value':1},open('result.json','w'))
        """.replace('SECRET',repr(os.fsencode(secret)))
        self.assert_objective(source,1)

    def test_dynamic_symlink_cannot_escape_even_for_a_native_reader(self):
        secret=Path(self.home)/'private.txt'
        secret.write_text('private input')
        self.assert_objective("""
            import os,ctypes,json
            getattr(os,'symlink')(SECRET,'scratch/link')
            libc=ctypes.CDLL(None,use_errno=True)
            assert libc.open(b'scratch/link',0)==-1
            json.dump({'status':'optimal','objective_value':1},open('result.json','w'))
        """.replace('SECRET',repr(str(secret))),1)

    def test_scratch_has_an_aggregate_size_limit(self):
        self.assert_objective("""
            import json
            block=b'x'*(1024*1024)
            denied=False
            try:
                for name in ('first','second'):
                    with open('scratch/'+name,'wb') as f:
                        for i in range(40): f.write(block)
            except OSError:
                denied=True
            json.dump({'status':'optimal','objective_value':int(denied)},open('result.json','w'))
        """,1)

    def test_threads_still_work(self):
        self.assert_objective("""
            import threading,json
            values=[]
            thread=threading.Thread(target=lambda:values.append(3))
            thread.start()
            thread.join()
            json.dump({'status':'optimal','objective_value':values[0]},open('result.json','w'))
        """,3)

    def test_missing_isolation_fails_closed(self):
        with patch('or_harness.execution.sandbox.command', side_effect=OSError('isolation unavailable')):
            r = self.execute("open('result.json','w').write('{}')")
        receipt = r.execution_features['technical_outcome']
        self.assertFalse(receipt['executed'])
        self.assertEqual(receipt['failure_stage'],'launch')
        self.assertNotIn('solver_runtime_s',r.cost.measured_dims())

    def test_memory_inside_a_runtime_mount_is_rejected_before_launch(self):
        p=self.plan()
        with patch('or_harness.execution.sandbox.runtime_paths',return_value=[str(self.h.home)]):
            r=self.execute('pass',p)
        receipt=r.execution_features['technical_outcome']
        self.assertFalse(receipt['process_started'])
        self.assertFalse(receipt['executed'])
        self.assertEqual(receipt['failure_stage'],'submission')
        self.assertIn('runtime directories',r.failures[0].error)

    def test_numpy_and_pandas_readers_share_boundary(self):
        if not all(importlib.util.find_spec(p) for p in ('numpy','pandas')):
            self.skipTest('optional numpy/pandas are not installed in this interpreter')
        self.resource('data.csv','x\n2\n3\n')
        hidden = self.work / 'hidden.csv'
        hidden.write_text('x\n999\n')
        self.assert_objective("""
            import numpy as np,pandas as pd,json
            from pathlib import Path
            path=Path('inputs')/'data.csv'
            a=pd.read_csv(path)['x'].sum()
            b=np.loadtxt(path,delimiter=',',skiprows=1).sum()
            for reader in (pd.read_csv,np.fromfile):
                try: reader(HIDDEN)
                except OSError: pass
                else: raise AssertionError('reader escaped the authorized inputs')
            json.dump({'status':'optimal','objective_value':int(a+b)},open('result.json','w'))
        """.replace('HIDDEN',repr(str(hidden))),10)


class TestCompositeFileStrategy(FileCase):
    def test_lp_reduction_milp_and_full_observation_loop(self):
        if importlib.util.find_spec('scipy') is None:
            self.skipTest('optional scipy.optimize is not installed in this interpreter')
        self.resource('columns.csv','cost,r0,r1,r2\n1,1,1,0\n1,0,1,1\n1,1,0,1\n5,1,0,0\n')
        self.resource('demand.json','[1,1,1]')
        method={'name':'LP and reduced-cost fixing then MILP','steps':[
            'Read the original column costs and coverage from files.',
            'Solve the LP relaxation and obtain dual prices and a lower bound.',
            'Construct a feasible cover, fix columns using a valid reduced-cost bound.',
            'Solve the reduced MILP and return an original-problem cover.']}
        self.provider.payload['capability_gain']={'decision':1,
            'claim':'Reduced costs and a feasible upper bound may certify safe column fixing.',
            'basis':['candidate LP, bound and reduction steps'],
            'applies_to':['binary covering with nonnegative costs'],
            'verification_conditions':[{'condition':'Check bound, fixed columns and original coverage.',
                'check_basis':'captured script/result and original-task check'}]}
        plan=self.h.plan_next(self.task,'ep',candidates=[{'solver':'python','method':method}])
        prediction=plan['selected_prediction_id']
        self.assertIsNotNone(prediction,plan)
        source="""
            import csv,json,os,itertools
            import numpy as np
            from scipy.optimize import linprog,milp,Bounds,LinearConstraint
            from pathlib import Path
            with (Path('inputs')/'columns.csv').open() as f: rows=list(csv.DictReader(f))
            with open(Path('inputs')/'demand.json') as f: demand=np.array(json.load(f))
            c=np.array([float(r['cost']) for r in rows])
            A=np.array([[float(r['r'+str(i)]) for r in rows] for i in range(3)])
            lp=linprog(c,A_ub=-A,b_ub=-demand,bounds=(0,None),method='highs',options={'threads':1})
            assert lp.success
            # A small feasible upper-bound construction is part of THIS strategy.
            upper=min(c@np.array(x) for x in itertools.product((0,1),repeat=len(c)) if np.all(A@np.array(x)>=demand))
            reduced=c+A.T@lp.ineqlin.marginals
            keep=np.flatnonzero(lp.fun+reduced<=upper+1e-8)
            final=milp(c[keep],integrality=np.ones(len(keep)),bounds=Bounds(0,1),
                constraints=LinearConstraint(A[:,keep],demand,np.inf),options={'threads':1})
            assert final.success
            solution=np.zeros(len(c));solution[keep]=np.rint(final.x)
            variables={str(i):float(x) for i,x in enumerate(solution)}
            variables.update(lp_bound=float(lp.fun),n_kept=len(keep),min_cover=float((A@solution).min()))
            result={'status':'optimal','objective_value':float(c@solution),'objective_bound':float(final.fun),
                'variables':variables,'method_performed':METHOD,
                'diagnostics':{'lp_bound':float(lp.fun),'columns_fixed':int(len(c)-len(keep))}}
            result['method_performed']['action_id']=os.environ['OR_ACTION_ID']
            json.dump(result,open('result.json','w'))
        """.replace('METHOD',repr(method))
        r=self.execute(source,prediction)
        self.assertEqual(r.quality['objective'],2,[f.error for f in r.failures])
        variables=r.execution_features['solution_variables']
        self.assertEqual(variables['lp_bound'],1.5)
        self.assertEqual(variables['n_kept'],3)
        self.assertEqual(r.method_actual, {**method,'source':'script_reported'} if 'source' in (r.method_actual or {}) else method)
        self.assertEqual(r.execution_features['prediction_id'],prediction)
        check={'integer':{'variables':['0','1','2','3']},
            'recompute_objective':{'coefficients':{'0':1,'1':1,'2':1,'3':5}},
            'semantic_probe':[{'path':'execution_features.solution_variables.min_cover','min':1}],
            'g_attribution':{'method_fidelity':'faithful','failure_source':'unknown','acceptance_scope':'original_task',
                'acceptance_id':'synthetic-cover-v1','source':'test review of original coverage and exact MILP certificate',
                'assessment':'Captured code solves LP, safely fixes a high reduced-cost column and solves the reduced binary model; original coverage and objective are checked.',
                'evidence_refs':['artifacts.submitted_script','artifacts.workspace_result']}}
        checked=self.h.check_task_result(r.execution_id,check)
        self.assertEqual(checked['state'],'passed',checked)
        self.h.record(r)
        closed=self.h.close_episode(self.task['task_id'],'ep')
        evaluation=closed['evaluations'][0]
        self.assertEqual(evaluation['benefit']['observed'],1)
        self.assertEqual(evaluation['risk']['implementation_observation']['label'],0)
        self.assertEqual(len(closed['induction_schedule']['pending_ids']),1)
        material=self.h.induction_material(induction_id=closed['induction_schedule']['pending_ids'][0])
        self.assertIn(r.execution_id,json.dumps(material))
        self.assertEqual(len(self.h.actions.query(action_type='execute_strategy')),1)
        self.assertEqual(len(jobs(self.h)),1)
