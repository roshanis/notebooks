"""Explicit automatic checks; semantic correctness always requires separate review."""

import json
import re
import statistics
from decimal import Decimal, InvalidOperation


def check_splits(train, validation, test):
    groups = [train, validation, test]
    for field in ['id', 'family_id', 'question']:
        seen = set()
        for group in groups:
            values = {re.sub(r'\W+', ' ', r[field].lower()).strip() for r in group}
            if field in {'id', 'question'} and len(values) != len(group):
                raise ValueError(f'Duplicate {field} within a split')
            if seen & values:
                raise ValueError(f'Overlapping {field}: {sorted(seen & values)[:3]}')
            seen.update(values)
    # Compositional/global test questions may use any state as evidence; training
    # must never contain their aggregate answer. Ordinary state facts are disjoint.
    seen_states = set()
    for group in groups:
        states = {s for r in group if r.get('category') != 'compositional' for s in r.get('states', [])}
        if seen_states & states:
            raise ValueError(f'Overlapping state facts: {sorted(seen_states & states)}')
        seen_states.update(states)
    for record in train + validation:
        if record.get('category') == 'compositional' or 'All.' in json.dumps(record.get('expected_values', {})):
            raise ValueError('Aggregate answers are reserved for compositional testing')
    return {'train': len(train), 'validation': len(validation), 'test': len(test), 'overlap': False}


def parse_prediction(prediction):
    if isinstance(prediction, str):
        prediction = json.loads(prediction)
    required = {'answer', 'citations', 'abstain', 'values'}
    if not isinstance(prediction, dict) or set(prediction) != required:
        raise ValueError('Expected exactly answer, citations, abstain, values')
    if not isinstance(prediction['answer'], str) or not prediction['answer'].strip():
        raise ValueError('Answer must be nonempty text')
    if type(prediction['abstain']) is not bool:
        raise ValueError('Abstain must be boolean')
    if not isinstance(prediction['citations'], list) or not all(isinstance(c, str) for c in prediction['citations']):
        raise ValueError('Citations must be a list of strings')
    if not isinstance(prediction['values'], dict):
        raise ValueError('Values must be an object')
    for k, v in prediction['values'].items():
        if (not isinstance(k, str) or not isinstance(v, str)
                or not re.fullmatch(r'-?\d+(?:\.\d+)?', v) or not Decimal(v).is_finite()):
            raise ValueError('Numeric values must be finite decimal strings')
    return prediction


def score_prediction(prediction, gold, available_citations):
    has_numbers = bool(gold['expected_values'])
    result = {'format_valid': False, 'numeric_exact': False if has_numbers else None,
              'unexpected_numeric_fields': None, 'citation_ids_valid': False,
              'required_citations_present': False, 'abstention_correct': False,
              'semantic_correctness': None, 'citation_entailment': None, 'caveats_preserved': None}
    try:
        pred = parse_prediction(prediction)
    except (ValueError, TypeError, InvalidOperation, json.JSONDecodeError):
        return result
    expected = gold['expected_values']
    result.update(format_valid=True,
                  numeric_exact=set(pred['values']) == set(expected) and all(
                      Decimal(pred['values'][k]) == Decimal(v) for k, v in expected.items()),
                  unexpected_numeric_fields=bool(set(pred['values']) - set(expected)),
                  citation_ids_valid=set(pred['citations']).issubset(available_citations),
                  required_citations_present=set(gold['required_citations']).issubset(pred['citations']),
                  abstention_correct=pred['abstain'] == gold['expected_abstain'])
    if not has_numbers and not pred['values']:
        result['numeric_exact'] = None
    return result


def summarize(records):
    output = {'n': len(records), 'automatic_checks': {}, 'denominators': {}, 'by_category': {},
              'answer_accuracy': None, 'publication_ready': False,
              'manual_semantic_review': 'pending; automatic checks do not establish answer correctness'}
    fields = ['format_valid', 'numeric_exact', 'citation_ids_valid', 'required_citations_present', 'abstention_correct']
    for field in fields:
        values = [r['scores'][field] for r in records if r['scores'][field] is not None
                  and (field != 'numeric_exact' or r.get('numeric_applicable', True))]
        output['denominators'][field] = len(values)
        output['automatic_checks'][field] = sum(values) / len(values) if values else None
    for category in sorted({r['category'] for r in records}):
        subset = [r for r in records if r['category'] == category]
        output['by_category'][category] = {'n': len(subset)}
        for field in fields:
            values = [r['scores'][field] for r in subset if r['scores'][field] is not None
                      and (field != 'numeric_exact' or r.get('numeric_applicable', True))]
            output['by_category'][category][field] = {'passed': sum(values), 'evaluated': len(values),
                                                     'rate': sum(values) / len(values) if values else None}
    output['generation_errors'] = sum(r.get('error') is not None for r in records)
    output['successful_generations'] = len(records) - output['generation_errors']
    output['successful_generation_checks'] = {}
    for field in fields:
        values = [r['scores'][field] for r in records if not r.get('error') and r['scores'][field] is not None
                  and (field != 'numeric_exact' or r.get('numeric_applicable', True))]
        output['successful_generation_checks'][field] = {'passed': sum(values), 'evaluated': len(values),
                                                        'rate': sum(values) / len(values) if values else None}
    seconds = [r['generation_seconds'] for r in records if r.get('generation_seconds') is not None and not r.get('error')]
    output['latency_seconds'] = {'n': len(seconds), 'median': statistics.median(seconds) if seconds else None,
                                 'minimum': min(seconds) if seconds else None, 'maximum': max(seconds) if seconds else None}
    return output
