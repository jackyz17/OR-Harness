"""One bounded comparison pack for the normal induction entry point."""
from __future__ import annotations

import copy
import time
from collections import Counter

from or_harness.strategy.evidence_views import chars, atomic, compact_execution, short_rule, task_identity, base_partitions
from or_harness.world_model.conditions import put_value
from or_harness.world_model.induction_gate import meta_write, meta_read


def seal_budget(result, *, summary=None):
    """Count the actual compact JSON, including the CLI envelope if present."""
    budget = result["budget"]
    for _ in range(8):
        total = chars({"result": result, "summary": summary}) if summary is not None else chars(result)
        if budget.get("chars_used") == total:
            break
        budget["chars_used"] = total
    return result


def build_compact_material(harness, *, strategy_id=None, task_id=None, limit=None,
                           cursor=None, related_top_k=5, purpose="modelling", gate=None):
    from or_harness.world_model.maintenance import (
        _induction_material_budget, _material_cursor, _parse_material_cursor, _query_text_for, _joint_hplus_for,
    )
    if purpose not in ("modelling", "implementation"):
        raise ValueError("induction purpose must be modelling or implementation")
    budget_limit = max(1024, _induction_material_budget())
    all_records = [r for r in harness.bank.all() if r.source == "executed" and r.measurement_scope == "attempt"]
    identities = {r.execution_id:task_identity(harness,r) for r in all_records}
    partition = dict(zip(identities,base_partitions(list(identities.values()))))
    base_attempts = Counter(partition.values())
    scoped = [r for r in all_records if (task_id is None or r.task_id == str(task_id))
              and (strategy_id is None or r.strategy_id == str(strategy_id))]
    scoped.sort(key=lambda r: (r.created_at, r.execution_id), reverse=True)
    previous, last = {}, {}
    for record in reversed(scoped):
        previous[record.execution_id] = last.get(record.task_id)
        last[record.task_id] = record
    total = len(scoped)
    if cursor and (boundary := _parse_material_cursor(cursor)) is not None:
        scoped = [r for r in scoped if (r.created_at, r.execution_id) < boundary]
    eligible = scoped[:max(0, limit)] if limit is not None else scoped
    # Query only a few representatives before expanding records or knowledge.
    seeds = eligible[:3]
    query = _query_text_for(seeds, harness)
    retrieval = {"executions": [], "knowledge": [], "failure": None}
    if seeds and related_top_k > 0 and query.get("text"):
        try:
            from or_harness.strategy.vector_recall import recall_vectors
            hits = recall_vectors(harness, query["text"], top_k=min(24, max(related_top_k * 4, 6)),
                include_unverified=True, task_profile=seeds[0].profile_snapshot,
                exclude_execution_ids={r.execution_id for r in scoped}, diversify=True, per_task_max=2)
            retrieval["executions"] = hits.get("execution_evidence") or []
            retrieval["knowledge"] = hits.get("strategic_knowledge") or []
            if hits.get("degraded"):
                retrieval["failure"] = {"reason":"related discovery degraded", "detail":atomic(hits["degraded"],"orx doctor",300)}
        except Exception as exc:
            retrieval["failure"] = {"kind": type(exc).__name__, "reason": str(exc)[:240]}
    seed_bases = {partition[r.execution_id] for r in seeds}
    related = []
    for hit in retrieval["executions"]:
        record = harness.bank.get(str(hit.get("execution_id")))
        if record is not None:
            related.append((record, hit.get("similarity")))
    # Distinct source problems first; same-source retries remain controls.
    def constraint_kinds(record):
        return {str(c.get("kind") or c.get("type")) for c in (record.cir_snapshot or {}).get("constraints") or [] if isinstance(c,dict)}
    seed_constraints = set().union(*(constraint_kinds(r) for r in seeds)) if seeds else set()
    related.sort(key=lambda pair: (
        (partition.get(pair[0].execution_id) not in seed_bases if purpose=="implementation" else
         partition.get(pair[0].execution_id) in seed_bases),
        not bool(pair[0].failures) if purpose=="implementation" else -len(seed_constraints & constraint_kinds(pair[0])),
        -(pair[1] or 0)))
    selected, base_counts = [], Counter()
    for record, score in related:
        identity = task_identity(harness, record)
        group = partition.get(record.execution_id,identity["base_id"])
        if base_counts[group] >= 2:
            continue
        base_counts[group] += 1
        selected.append((record, score))
        if len(selected) >= related_top_k:
            break
    semantic_ids = {str(x.get("entry_id")) for x in retrieval["knowledge"]}
    evidence_ids = {r.execution_id for r in seeds}
    families = {r.profile_snapshot.family for r in seeds}
    entries, n_candidate_entries = harness.sbank.material_candidates(
        strategy_ids={r.strategy_id for r in seeds},execution_ids=evidence_ids,families=families,semantic_ids=semantic_ids,limit=8)
    attempts = Counter(r.task_id for r in scoped)
    out = {
        "version": "induction-material/2", "purpose": purpose, "induction_gate": gate,
        "material_id": "induction_material:" + "0" * 64,
        "count": 0, "total_completed": total, "n_distinct_tasks": 0,
        "cross_task_hint": {"n_distinct_tasks_in_batch": 0, "distinct_tasks_across_history": len({r.task_id for r in all_records})},
        "material": [], "related_history": {"enabled": related_top_k > 0, "executions": [], "knowledge": [],
            "failure": retrieval["failure"], "query_basis": copy.deepcopy(query),
            "execution_selection": {"n_distinct_tasks": len({r.task_id for r, _ in selected}),
                                    "n_distinct_base_problems": len({task_identity(harness,r)["base_id"] for r,_ in selected})}},
        "existing_knowledge": [], "comparison_table": [], "joint_hplus": _joint_hplus_for(harness,seeds),
        "check_states": {}, "task_chains": {},
        "memory_state": {"state": "both_banks_empty" if not all_records and not harness.sbank.count() else
            "strategic_bank_empty" if not harness.sbank.count() else "evidence_bank_empty" if not all_records else "both_banks_present",
            "evidence_bank_empty": not bool(all_records), "both_empty":not all_records and not harness.sbank.count()},
        "budget": {"chars_limit": budget_limit, "chars_used": 0, "truncated_by_budget": False,
            "truncated_by_limit": limit is not None and len(eligible) < len(scoped),
            "omitted_execution_ids": [], "omitted_knowledge_count": 0, "next_cursor": None},
        "guidance": {"structural_contrast":"Compare necessary constraint relations, actual methods and checked outcomes; similarity/family only find candidates.",
                     "key_step_explanation":"Write when/do/expect; choose the existing ID for a revision. Shown cases are derivation material.",
                     "boundary_check":"Test changed premises and an independent unseen case; same-base retries/seeds are controls."},
    }
    # Leave room for the normal CLI summary and its serialized envelope.
    out['joint_hplus']['role'] = 'pre-solve estimate; actual start judgement is induction_gate'
    out['joint_hplus']['items'] = [atomic(item,'orx inspect --bank capability --prediction '+str(item.get('prediction_id')),1200) for item in out['joint_hplus'].get('items',[])[:3]]
    out['related_history']['no_hits'] = None if retrieval['failure'] or related_top_k<=0 else not bool(selected or entries)
    if budget_limit<6000:
        out['related_history']['query_basis'] = {'basis':query.get('basis'),'missing':query.get('missing') or []}
        out['joint_hplus'] = {k:out['joint_hplus'][k] for k in ('n','n_expected','n_none') if k in out['joint_hplus']}
    payload_limit = max(800, budget_limit - min(500, budget_limit // 8))
    field_limit = max(150, min(2400, budget_limit // 10))

    def add_case(record, target, score=None):
        row = compact_execution(harness, record, field_limit=field_limit)
        prior = previous.get(record.execution_id)
        row['outcome']['code_hash'] = (record.solver or {}).get('code_hash')
        row['changes'] = {'previous_execution_id':prior.execution_id if prior else None,
                          'code_hash_changed': ((record.solver or {}).get('code_hash') != (prior.solver or {}).get('code_hash')) if prior else None}
        row.update(attempts_of_task=attempts.get(record.task_id, 1), independent_task=(None if row['identity']['basis']=='task_id_only' else base_attempts[partition[record.execution_id]]==1),
                   cursor=_material_cursor(record))
        if score is not None:
            row["discovery"] = {"similarity": score, "meaning": "candidate_for_comparison"}
        row_limit = min(6000,max(500,payload_limit//3))
        if chars(row)>row_limit:
            row = {k:row[k] for k in ('execution_id','task_id','identity','method','method_source','outcome','task_check_state','cost','cost_measured','measurement_scope','inspect_hint','cursor','attempts_of_task','independent_task','changes')}
            row['method'] = atomic(row['method'],row['inspect_hint']+' #method',max(100,row_limit//3))
            row['identity'] = {k:row['identity'].get(k) for k in ('base_id','basis')}
        relation = "current_task" if target is out["material"] else (
            "same_source_control" if partition[record.execution_id] in seed_bases else "cross_source_candidate")
        index = {"execution_id": record.execution_id, "relationship": relation,
                 "actual_method_ref": record.execution_id + ".method.actual", "check": row["task_check_state"]}
        target.append(row)
        out["comparison_table"].append(index)
        seal_budget(out)
        if chars(out) > payload_limit:
            target.pop(); out["comparison_table"].pop()
            out["budget"]["truncated_by_budget"] = True
            return False
        return True

    # Protect one seed, one cross-source case and a relevant old rule before
    # filling spare space with retries and other neighbours.
    if seeds:
        if not add_case(seeds[0], out["material"]):
            r=seeds[0]
            out['material'].append({'execution_id':r.execution_id,'task_id':r.task_id,'cursor':_material_cursor(r),
                'identity':identities[r.execution_id], 'task_check_state':compact_execution(harness,r)['task_check_state'],
                'inspect_hint':f'orx inspect --bank experience --execution {r.execution_id}', 'view':'whole_record_by_reference'})
    if selected:
        add_case(selected[0][0], out["related_history"]["executions"], selected[0][1])
    if len(selected)>1:
        add_case(selected[1][0], out["related_history"]["executions"], selected[1][1])
    for entry, reasons in entries:
        row = short_rule(entry, field_limit=field_limit)
        row["related_by"] = reasons
        uses = [r for r in all_records if entry.entry_id in
                ((r.execution_features.get('used_entries') or {}).get('known_entry_ids') or [])]
        effects = (harness.store.conn.execute(
            "SELECT COUNT(*) FROM rule_events WHERE entry_id=? AND kind='effect'", (entry.entry_id,)).fetchone()[0]
            if entry.rule else len(entry.effect_attribution))
        row['usage'] = {'n_adoptions':len(uses), 'n_known_tasks':len({r.task_id for r in uses}),
                        'n_effect_attributions':effects, 'unattributed':bool(uses) and not effects,
                        'detail_ref':row['detail_ref']}
        if entry.rule and entry.rule.get('candidate_version') and entry.rule.get('candidate_version')!=entry.rule.get('view_version'):
            row['candidate'] = short_rule(harness.sbank.get(entry.entry_id,version='candidate'),field_limit=field_limit)
        out["existing_knowledge"].append(row)
        seal_budget(out)
        if chars(out) > payload_limit:
            out["existing_knowledge"].pop();out["budget"]["truncated_by_budget"] = True
    for record in eligible[1:]:
        if not add_case(record, out["material"]):
            break  # contiguous pagination, no silently skipped execution
    for record, score in selected[2:]:
        add_case(record, out["related_history"]["executions"], score)
    def refresh():
        retained = {r['execution_id'] for r in out['material']}
        omitted = [r for r in scoped if r.execution_id not in retained]
        displayed = out['material']+out['related_history']['executions']
        out['count'] = len(retained)
        out['n_distinct_tasks'] = len({r['task_id'] for r in out['material']})
        out['tasks'] = sorted({r['task_id'] for r in out['material']})
        out['task_chains'] = {task:[r['execution_id'] for r in out['material'] if r['task_id']==task]
                              for task in out['tasks']}
        out['cross_task_hint'].update(n_distinct_tasks_in_batch=out['n_distinct_tasks'],
            n_distinct_base_problems_shown=len({partition[r['execution_id']] for r in displayed}),
            note='1 task only: conditional_fact remains derivation evidence.' if out['n_distinct_tasks']==1 else 'Look for a mechanism that recurs. Task counts are facts; source problems, not retries, supply independent support.')
        out['budget'].update(omitted_execution_ids=[r.execution_id for r in omitted[:3]],
            omitted_execution_count=len(omitted), omitted_knowledge_count=n_candidate_entries-len(out['existing_knowledge']),
            next_cursor=out['material'][-1]['cursor'] if omitted and out['material'] else None)
        states=Counter(r['task_check_state'] for r in out['material'])
        out['check_states']={'passed':states['passed'],'failed':states['failed'],
                             'never_checked_or_insufficient':len(retained)-states['passed']-states['failed']}
        out['related_history']['knowledge']=[{'entry_id':r['entry_id'],'body_ref':f"existing_knowledge[{r['entry_id']}]"}
            for r in out['existing_knowledge'] if 'semantic' in r['related_by']]
        if 'comparison_table' in out:
            out['comparison_table']=[{'execution_id':r['execution_id'],'check':r['task_check_state'],
                'relationship':'current_task' if r['execution_id'] in retained else
                'same_source_control' if partition[r['execution_id']] in seed_bases else 'cross_source_candidate'} for r in displayed]
        seal_budget(out)

    refresh()
    # Recompute the cursor and displayed IDs AFTER final metadata trimming;
    # no budget eviction may create a pagination gap or a fictitious read.
    while chars(out)>payload_limit:
        if out["existing_knowledge"]:
            out["existing_knowledge"].pop()
        elif out["related_history"]["executions"]:
            out["related_history"]["executions"].pop()
        elif len(out['material'])>1:
            out['material'].pop()
        else:
            out['budget']['truncated_by_budget']=True
            for key in ('guidance','comparison_table','task_chains','joint_hplus'):
                out.pop(key,None)
            out['related_history']['query_basis']={'detail_ref':'orx inspect --bank experience --task '+str(task_id),'basis':query.get('basis')}
            if out['material']:
                row=out['material'][0]
                out['material'][0]={k:row[k] for k in ('execution_id','task_id','cursor','task_check_state','inspect_hint')}
                out['material'][0]['view']='whole_record_by_reference'
            refresh()
            break
        out["budget"]["truncated_by_budget"] = True
        refresh()
    displayed = out['material']+out['related_history']['executions']
    session = {'task_id':task_id,'purpose':purpose,'shown_execution_ids':[r['execution_id'] for r in displayed],
        'prior_material_ref':meta_read(harness.store,'induction_material:last:'+str(task_id)) if task_id is not None else None,
        'identities':{r['execution_id']:identities[r['execution_id']] for r in displayed},'created_at':time.time()}
    with harness.store.transaction() as conn:
        material_id=put_value(conn,harness.store,'induction_material',session)
        if task_id is not None:
            meta_write(harness.store,'induction_material:last:'+str(task_id),material_id)
    out['material_id']=material_id
    seal_budget(out)
    if chars(out)>budget_limit:
        # The minimum configuration still returns one reachable row, with
        # accurate whole-envelope accounting, instead of a stalled cursor.
        for key in ('cross_task_hint','memory_state','related_history','check_states','purpose','n_distinct_tasks','total_completed'):
            out.pop(key,None)
        seal_budget(out)
    return out
