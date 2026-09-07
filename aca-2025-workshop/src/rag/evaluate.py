"""Frozen-label adapters and retrieval metrics. No labels enter the search API."""
import json
from pathlib import Path
from src.rag.core import digest


def questions_from_rows(rows):
    questions=[]
    for row in rows:
        ns=row['namespace'];filters=dict(namespace=ns,year=2025)
        if ns=='plans':
            for key in ['plan_ids','state','age','tobacco','rating_area','date']:
                if row.get('context',{}).get(key) is not None:filters[key]=row['context'][key]
        else:
            # Resolve only from the user's words, never from expected answer keys.
            import re
            from src.data import STATE_NAMES
            remaining=row['question'].lower();states=[]
            for state,name in sorted(STATE_NAMES.items(),key=lambda item:-len(item[1])):
                pattern=r'\b'+re.escape(name.lower())+r'\b'
                if re.search(pattern,remaining):
                    states.append(state);remaining=re.sub(pattern,' ',remaining)
            if len(states)==1:filters['state']=states[0]
        questions.append(dict(id=row['id'],namespace=ns,question=row['question'],filters=filters,
                              reference_answer=row.get('reference_answer',row.get('answer','')),
                              expected_values=row.get('expected_values',{}),expected_action=row.get('expected_action'),
                              required_citations=row.get('required_citations',[])))
    return questions


def retrieval_ablation(retriever,questions,output,k=8):
    # Uses reference citation sets as partial source judgments; not complete relevance labels.
    output=Path(output);output.mkdir(parents=True,exist_ok=False);rows=[]
    for q in questions:
        required=set(q.get('required_citations',[]))
        if not required:continue
        for method in ['bm25','semantic','graph','hybrid']:
            result=retriever.search(q['question'],q['filters'],method,k)
            found=set();first=None
            for rank,r in enumerate(result['results'],1):
                d=r['document'];ids={d['source_id'],d['record_id'],d['id']}
                hits=required&ids
                if hits and first is None:first=rank
                found|=hits
            # Report unresolvable expected IDs rather than treating partial judgments as exhaustive truth.
            known={v for d in retriever.docs for v in [d['source_id'],d['record_id'],d['id']]}
            resolvable=required&known
            rows.append(dict(id=q['id'],namespace=q['namespace'],method=method,seconds=result['seconds'],
                             reference_count=len(required),resolvable_references=len(resolvable),
                             partial_source_recall=len(found)/len(resolvable) if resolvable else None,
                             reciprocal_rank=1/first if first else 0.,retrieved=[r['document']['id'] for r in result['results']]))
    (output/'retrieval.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summary=[]
    for ns in ['plans','enrollment']:
        for method in ['bm25','semantic','graph','hybrid']:
            group=[r for r in rows if r['namespace']==ns and r['method']==method and r['partial_source_recall'] is not None]
            if group:summary.append(dict(namespace=ns,method=method,n=len(group),partial_source_recall_at_k=sum(r['partial_source_recall'] for r in group)/len(group),
                                          mrr=sum(r['reciprocal_rank'] for r in group)/len(group),mean_seconds=sum(r['seconds'] for r in group)/len(group)))
    report=dict(k=k,questions_hash=digest(questions),config=retriever.config,summary=summary,
                caveat='Existing citation sets are incomplete relevance judgments. Unresolvable references excluded explicitly. Graph uses lexical seeds. No weights tuned on final test.')
    (output/'summary.json').write_text(json.dumps(report,indent=2));return report
