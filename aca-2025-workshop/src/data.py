"""Strict CMS CSV reader. All calculations use Decimal and preserve source status."""

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_NAMES = dict(item.split(':', 1) for item in (
    'AK:Alaska|AL:Alabama|AR:Arkansas|AZ:Arizona|CA:California|CO:Colorado|CT:Connecticut|'
    'DC:District of Columbia|DE:Delaware|FL:Florida|GA:Georgia|HI:Hawaii|IA:Iowa|ID:Idaho|'
    'IL:Illinois|IN:Indiana|KS:Kansas|KY:Kentucky|LA:Louisiana|MA:Massachusetts|MD:Maryland|'
    'ME:Maine|MI:Michigan|MN:Minnesota|MO:Missouri|MS:Mississippi|MT:Montana|NC:North Carolina|'
    'ND:North Dakota|NE:Nebraska|NH:New Hampshire|NJ:New Jersey|NM:New Mexico|NV:Nevada|'
    'NY:New York|OH:Ohio|OK:Oklahoma|OR:Oregon|PA:Pennsylvania|RI:Rhode Island|SC:South Carolina|'
    'SD:South Dakota|TN:Tennessee|TX:Texas|UT:Utah|VA:Virginia|VT:Vermont|WA:Washington|'
    'WI:Wisconsin|WV:West Virginia|WY:Wyoming'
).split('|'))

# Deliberately bounded first-release metrics. Raw originals preserve all columns.
METRICS = {
    'Cnsmr': ('plan selections', 'people', 3),
    'New_Cnsmr': ('new consumers', 'people', 3),
    'Tot_Renrl': ('total re-enrollees', 'people', 4),
    'Actv_Renrl': ('active re-enrollees', 'people', 4),
    'Auto_Renrl': ('automatic re-enrollees', 'people', 4),
    'Avg_Prm': ('average monthly premium before APTC', 'USD/person/month', 6),
    'Avg_Prm_Aftr_APTC': ('average monthly premium after APTC', 'USD/person/month', 6),
    'APTC_Cnsmr': ('consumers with APTC', 'people', 7),
    'Actv_Renrl_Sw': ('active re-enrollees who switched plans', 'people', 4),
    'BHP_Enrlmnt': ('Basic Health Program enrollments', 'people', 14),
}


@dataclass(frozen=True)
class Cell:
    value: Decimal | None
    status: str
    raw: str

    def as_dict(self):
        return {'value': str(self.value) if self.value is not None else None,
                'status': self.status, 'raw': self.raw}


def parse_cell(raw):
    if not isinstance(raw, str):
        raise ValueError('Missing or malformed CSV cell')
    text = raw.strip()
    statuses = {'': 'missing', '+': 'not_applicable', 'NR': 'not_reported',
                '*': 'suppressed', 'N/A': 'unavailable'}
    if text in statuses:
        return Cell(None, statuses[text], raw)
    if not re.fullmatch(r'\$?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?', text):
        raise ValueError(f'Invalid numeric source value: {raw!r}')
    return Cell(Decimal(text.replace('$', '').replace(',', '')), 'reported', raw)


class EnrollmentTable:
    def __init__(self, states, aggregates):
        self.states = states
        self.aggregates = aggregates

    @classmethod
    def from_csv(cls, stream, require_full=True):
        reader = csv.DictReader(stream)
        required = {'State_Abrvtn', 'Pltfrm', 'Cnsmr'}
        if require_full:
            required.update(METRICS)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError('Missing required CMS columns')
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('Duplicate CSV headers')
        states, aggregates = {}, {}
        for line, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'CSV column count mismatch at row {line}')
            state, platform = row['State_Abrvtn'].strip(), row['Pltfrm'].strip()
            aggregate = state == 'Total'
            if platform not in ({'HC.gov', 'SBE', 'All'} if aggregate else {'HC.gov', 'SBE'}):
                raise ValueError(f'Unknown platform: {platform}')
            if not aggregate and state not in STATE_NAMES:
                raise ValueError(f'Unknown state: {state}')
            cells = {k: parse_cell(v) for k, v in row.items() if k not in ('State_Abrvtn', 'Pltfrm')}
            for metric, cell in cells.items():
                if metric in METRICS and METRICS[metric][1] == 'people' and cell.value is not None:
                    if cell.value != cell.value.to_integral_value():
                        raise ValueError(f'Non-integral count in {state}/{metric}')
            target, key = (aggregates, platform) if aggregate else (states, state)
            if key in target:
                raise ValueError(f'Duplicate source row: {state}/{platform}')
            target[key] = {'platform': platform, 'cells': cells, 'source_row': line}
        if require_full:
            if set(states) != set(STATE_NAMES) or set(aggregates) != {'HC.gov', 'SBE', 'All'}:
                raise ValueError('Expected 51 jurisdictions and three aggregate rows')
            counts = {p: sum(r['platform'] == p for r in states.values()) for p in ['HC.gov', 'SBE']}
            if counts != {'HC.gov': 31, 'SBE': 20}:
                raise ValueError(f'Unexpected 2025 platform coverage: {counts}')
        result = cls(states, aggregates)
        if require_full:
            for platform in ['All', 'HC.gov', 'SBE']:
                if result.total('Cnsmr', platform).value != result.published_total('Cnsmr', platform).value:
                    raise ValueError(f'Plan-selection reconciliation failed: {platform}')
        return result

    @classmethod
    def load(cls, root=ROOT):
        verify_sources(root)
        with (Path(root) / 'data/raw/state.csv').open(encoding='utf-8-sig', newline='') as stream:
            return cls.from_csv(stream)

    def lookup(self, state, metric):
        if metric not in METRICS:
            raise ValueError(f'Unsupported metric: {metric}')
        state = state.upper()
        if state not in self.states or metric not in self.states[state]['cells']:
            raise ValueError(f'Unavailable state/metric: {state}/{metric}')
        return self.states[state]['cells'][metric]

    def total(self, metric, platform='All'):
        if metric not in METRICS or METRICS[metric][1] != 'people':
            raise ValueError('Only supported counts may be summed; use published averages')
        if platform not in {'All', 'HC.gov', 'SBE'}:
            raise ValueError('Unknown platform')
        cells = [self.lookup(s, metric) for s, r in self.states.items()
                 if platform == 'All' or r['platform'] == platform]
        if not cells or any(c.value is None for c in cells):
            return Cell(None, 'incomplete', '')
        return Cell(sum((c.value for c in cells), Decimal(0)), 'calculated', '')

    def published_total(self, metric, platform='All'):
        if metric not in METRICS or platform not in self.aggregates:
            raise ValueError('Unsupported metric or aggregate')
        return self.aggregates[platform]['cells'][metric]

    def rank(self, metric, n=5):
        if not isinstance(n, int) or not 1 <= n <= len(self.states):
            raise ValueError('Invalid ranking size')
        values = [(s, self.lookup(s, metric).value) for s in self.states]
        if any(v is None for _, v in values):
            raise ValueError('Cannot rank a metric with unavailable state values')
        return sorted(values, key=lambda x: (-x[1], x[0]))[:n]


def verify_sources(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / 'data/source_manifest.json').read_text())
    for source in manifest['files']:
        path = (root / source['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Source path escapes project')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != source['sha256']:
            raise ValueError(f'Source checksum mismatch: {source["path"]}')
    return manifest
