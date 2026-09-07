"""Plan-answer evaluation: structural checks plus mandatory semantic review."""
import hashlib
import json
import re
from src.plan_data import ROOT


def check_plan_splits(train, validation, test):
    seen_ids, seen_questions, seen_groups = set(), set(), set()
    for records in [train, validation, test]:
        groups = {g for r in records for g in r['groups']}
        ids = [r['id'] for r in records]
        questions = [re.sub(r'\W+', ' ', r['question'].lower()).strip() for r in records]
        if len(set(ids)) != len(ids) or len(set(questions)) != len(questions):
            raise ValueError('Duplicate questions or IDs within split')
        if seen_ids & set(ids) or seen_questions & set(questions) or seen_groups & groups:
            raise ValueError('Question or issuer/state family leakage across splits')
        seen_ids.update(ids)
        seen_questions.update(questions)
        seen_groups.update(groups)


def validate_plan_datasets(root=ROOT):
    folder = root / 'eval/plans_2025'
    manifest = json.loads((folder / 'split_manifest.json').read_text())
    for relative, expected in manifest['sha256'].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen plan dataset changed: ' + relative)
    datasets = [[json.loads(line) for line in (folder / name).read_text().splitlines() if line.strip()]
                for name in ['training.jsonl', 'validation.jsonl', 'questions.jsonl']]
    check_plan_splits(*datasets)
    return tuple(datasets)


def score_plan_prediction(prediction, gold, available_citations):
    scores = dict.fromkeys(['format_valid', 'action_correct', 'citation_ids_valid', 'required_citations_present'], False)
    scores.update(semantic_correctness=None, completeness=None, citation_entailment=None,
                  caveats_preserved=None, helpful_clarification=None)
    try:
        p = json.loads(prediction) if isinstance(prediction, str) else prediction
        if not isinstance(p, dict) or set(p) != {'answer', 'action', 'citations'}:
            return scores
        if not isinstance(p['answer'], str) or not p['answer'].strip() or p['action'] not in {'answer', 'clarify', 'insufficient_evidence'}:
            return scores
        if not isinstance(p['citations'], list) or not all(isinstance(c, str) for c in p['citations']):
            return scores
        scores.update(format_valid=True, action_correct=p['action'] == gold['expected_action'],
                      citation_ids_valid=set(p['citations']).issubset(available_citations),
                      required_citations_present=set(gold['required_citations']).issubset(p['citations']))
    except (ValueError, TypeError):
        pass
    return scores


def summarize_plan_results(records):
    fields = ['format_valid', 'action_correct', 'citation_ids_valid', 'required_citations_present']
    def rates(rows):
        return {'n': len(rows), **{k: sum(r['scores'][k] for r in rows) / len(rows) if rows else None for k in fields}}
    labeled = [r for r in records if r.get('expected_action')]
    actions = sorted({r['expected_action'] for r in labeled})
    by_action = {a: rates([r for r in labeled if r['expected_action'] == a]) for a in actions}
    return {'n': len(records), 'automatic_checks': rates(records),
            'score_scope': 'Structural diagnostics only; never an overall answer-quality pass',
            'by_expected_action': by_action,
            'balanced_action_accuracy': sum(v['action_correct'] for v in by_action.values()) / len(by_action) if by_action else None,
            'majority_action_baseline': max((sum(r['expected_action'] == a for r in labeled) for a in actions), default=0) / len(labeled) if labeled else None,
            'by_family': {f: rates([r for r in records if r.get('family_id', r['category']) == f]) for f in sorted({r.get('family_id', r['category']) for r in records})},
            'by_issuer_state': {g: rates([r for r in records if g in r.get('groups', [])]) for g in sorted({g for r in records for g in r.get('groups', [])})},
            'by_category': {c: rates([r for r in records if r['category'] == c]) for c in sorted({r['category'] for r in records})},
            'generation_errors': sum(r.get('error') is not None for r in records),
            'answer_accuracy': None, 'publication_ready': False,
            'manual_review': 'Required for every answer. Structural success is not proof of coverage, price, or citation correctness.'}
