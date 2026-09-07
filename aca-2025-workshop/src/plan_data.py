"""Strict lookups over the versioned 2025 plan snapshot. No network or model calls."""
import hashlib
import json
import re
from datetime import date as Date
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def money(value):
    """Only bare source amounts are numbers; qualifiers and missing values survive."""
    cleaned = str(value).strip().replace('$', '').replace(',', '')
    return Decimal(cleaned) if re.fullmatch(r'\d+(?:\.\d+)?', cleaned) else None


class PlanCatalog:
    def __init__(self, data):
        self.data = data
        self.plans = data['plans']

    @classmethod
    def load(cls, root=ROOT):
        return cls(json.loads((root / 'data/plans_2025/catalog.json').read_text()))

    def plan(self, plan_id):
        if plan_id not in self.plans:
            raise ValueError('Exact plan variant is not in this selected 2025 snapshot')
        return self.plans[plan_id]

    def cost_fields(self, plan_id):
        a = self.plan(plan_id)['attributes']
        ded = 'TEHB' if a.get('MedicalDrugDeductiblesIntegrated') == 'Yes' else 'MEHB'
        moop = 'TEHB' if a.get('MedicalDrugMaximumOutofPocketIntegrated') == 'Yes' else 'MEHB'
        return {'deductible': a.get(ded + 'DedInnTier1Individual', ''),
                'deductible_field': ded + 'DedInnTier1Individual',
                'out_of_pocket_max': a.get(moop + 'InnTier1IndividualMOOP', ''),
                'out_of_pocket_field': moop + 'InnTier1IndividualMOOP',
                'deductible_integrated': a.get('MedicalDrugDeductiblesIntegrated', ''),
                'drug_deductible': a.get('DEHBDedInnTier1Individual', '')}

    def benefit(self, plan_id, name):
        found = [r for r in self.plan(plan_id)['benefits'] if r['BenefitName'] == name]
        if len(found) != 1:
            raise ValueError('Benefit is missing or ambiguous; do not infer coverage')
        return found[0]

    def availability(self, plan_id, county_fips=None, zip_code=None):
        rows = self.plan(plan_id)['service_areas']
        if not county_fips:
            return 'needs_county'
        matches = [r for r in rows if r['County'] == county_fips or r['CoverEntireState'] == 'Yes']
        if not matches:
            return 'not_in_service_area'
        if any(r['CoverEntireState'] == 'Yes' or r['PartialCounty'] == 'No' for r in matches):
            return 'available'
        if not zip_code:
            return 'needs_zip'
        if any(zip_code in re.findall(r'\b\d{5}\b', r.get('ZipCodes', '')) for r in matches):
            return 'available'
        # CMS warns long ZIP lists can be truncated; absence is not proof of exclusion.
        return 'unverified_zip'

    def rate(self, plan_id, *, age=None, tobacco=None, rating_area=None, date=None):
        if type(age) is not int or type(tobacco) is not bool or not rating_area or not date:
            raise ValueError('Rate needs individual age, tobacco status, rating area and effective date')
        if Date.fromisoformat(date).year != 2025:
            raise ValueError('Only historical 2025 rates are supported')
        a = self.plan(plan_id)['attributes']
        rows = [r for r in self.plan(plan_id)['rates']
                if r['PlanId'] == a['StandardComponentId'] and r['BusinessYear'] == '2025'
                and r['StateCode'] == a['StateCode'] and r['IssuerId'] == a['IssuerId']
                and r['Age'] == str(age) and r['RatingAreaId'] == rating_area
                and r['RateEffectiveDate'] <= date <= r['RateExpirationDate']]
        if len(rows) != 1:
            raise ValueError('Rate unavailable or conflicting; cannot select an arbitrary row')
        row = rows[0]
        field = 'IndividualTobaccoRate' if tobacco and row['Tobacco'] != 'No Preference' else 'IndividualRate'
        value = money(row.get(field, ''))
        if value is None or value >= 999999:
            raise ValueError('Rate not available as a usable individual price')
        return row, value


def verify_plan_sources(root=ROOT):
    manifest = json.loads((root / 'data/plans_2025/source_manifest.json').read_text())
    for item in manifest['files']:
        actual = hashlib.sha256((root / item['path']).read_bytes()).hexdigest()
        if actual != item['sha256']:
            raise ValueError('Plan source checksum mismatch: ' + item['path'])
    return manifest
