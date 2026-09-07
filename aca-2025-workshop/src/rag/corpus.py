"""Build source-only chunks; training/reference files are never crawled."""
import csv
import io
import json
from pathlib import Path
from src.rag.core import digest
from src.data import STATE_NAMES, METRICS

ROOT=Path(__file__).resolve().parents[2]
SOURCE_PATHS=['data/source_manifest.json','data/reporting_periods.json','data/raw/state.csv',
              'data/documents/pages.jsonl','data/plans_2025/source_manifest.json',
              'data/plans_2025/catalog.json','data/plans_2025/sbc_pages.json']


def source_records(root=ROOT):
    from src.evidence import EvidenceEngine
    from src.plan_data import PlanCatalog,verify_plan_sources
    from src.plan_evidence import clean_fields,BENEFIT_FIELDS
    engine=EvidenceEngine.load(root);verify_plan_sources(root);records=[]
    def add(id,text,namespace,kind,source_id,locator,url='',state='',plan_ids=None,links=None,**extra):
        records.append(dict(id=id,text=text,namespace=namespace,year=2025,kind=kind,source_id=source_id,
                            locator=locator,url=url,state=state,plan_ids=plan_ids or [],links=links or [],**extra))
    pages=[json.loads(line) for line in (root/'data/documents/pages.jsonl').read_text().splitlines()]
    for p in pages:
        add('page:'+p['id'],p['text'],'enrollment','page',p['source_id'],f'page {p["page"]}',p['url'])
    for state in list(STATE_NAMES)+['All']:
        for metric,(_,_,page) in METRICS.items():
            id,text,_=engine.table_evidence(state,metric)
            text=STATE_NAMES.get(state,'National')+'\n'+text
            links=[dict(target=f'page:definitions-p{page}',relation='defined_by',source=id),
                   dict(target='page:faqs-p1',relation='reporting_scope',source=id)]
            add(id,text,'enrollment','observation','state_csv',id,engine.sources['state_csv']['url'],state=state,links=links)
    catalog=PlanCatalog.load(root)
    sources=json.loads((root/'data/plans_2025/source_manifest.json').read_text())['files']
    urls={Path(s['path']).name:s.get('url','') for s in sources}
    for pid,plan in sorted(catalog.plans.items()):
        a=plan['attributes'];state=a['StateCode'];head=f'2025 {state} {pid} {a["PlanMarketingName"]}\n'
        identity=clean_fields(a,['PlanId','PlanMarketingName','IssuerMarketPlaceMarketingName','StateCode','MetalLevel',
                  'CSRVariationType','PlanType','IsReferralRequiredForSpecialist','SpecialistRequiringReferral','MultipleInNetworkTiers'])
        cost=dict(identity,**catalog.cost_fields(pid))
        costid='plan:'+pid+':cost'
        source=a['_source']
        add(costid,head+json.dumps(cost)+'\nMissing or Not Applicable is not zero. Preserve network tiers and deductible integration.',
            'plans','attributes',source['citation'],f'{source["member"]} record {source["record"]}',urls[source['archive']],state,[pid])
        for kind,key in [('benefit','benefits'),('rate','rates'),('service_area','service_areas')]:
            for ordinal,row in enumerate(plan[key]):
                s=row['_source'];id=f'plan:{pid}:{s["citation"]}:{ordinal}'
                if kind=='rate':
                    # A CMS row can contain both tobacco columns. Split their
                    # meanings explicitly, rather than treating the flag as a
                    # single mutually exclusive row-level tobacco category.
                    for tobacco in [False,True]:
                        field='IndividualTobaccoRate' if tobacco and row['Tobacco']!='No Preference' else 'IndividualRate'
                        rid=id+(':tobacco' if tobacco else ':non-tobacco')
                        value=row.get(field,'')
                        payload={k:row[k] for k in ['PlanId','Age','RatingAreaId','RateEffectiveDate','RateExpirationDate','Tobacco']}
                        payload.update(selected_tobacco=tobacco,source_rate_column=field,gross_monthly_individual_rate=value,
                                       note='Before subsidies. Missing/unavailable is not zero. Rating area does not establish county eligibility.')
                        add(rid,head+json.dumps(payload),'plans','rate',s['citation'],f'{s["member"]} record {s["record"]}, column {field}',
                            urls[s['archive']],state,[pid],links=[dict(target=costid,relation='plan_cost_rules',source=rid)],
                            profile=dict(age=row['Age'],rating_area=row['RatingAreaId'],tobacco=tobacco,start=row['RateEffectiveDate'],end=row['RateExpirationDate']))
                    continue
                fields=clean_fields(row,BENEFIT_FIELDS) if kind=='benefit' else {k:v for k,v in row.items() if k!='_source' and v not in ('',None,'nan')}
                note=''
                if kind=='rate':note=' Gross monthly individual rates before subsidies. Select age, tobacco, rating area and effective date exactly; not a personal quote.'
                if kind=='service_area':note=' Service-area membership is not a verified county-to-rating-area mapping.'
                add(id,head+json.dumps(fields)+note,'plans',kind,s['citation'],f'{s["member"]} record {s["record"]}',
                    urls[s['archive']],state,[pid],links=[dict(target=costid,relation='plan_cost_rules',source=id)],
                    profile=dict(age=row.get('Age'),rating_area=row.get('RatingAreaId'),start=row.get('RateEffectiveDate'),end=row.get('RateExpirationDate')))
    for item in json.loads((root/'data/plans_2025/sbc_pages.json').read_text()):
        pid=item['plan_id'];a=catalog.plans[pid]['attributes']
        for page in item['pages']:
            id=f'sbc:{pid}:p{page["page"]}'
            add(id,f'{pid} {a["PlanMarketingName"]}\nSBC extracted page text; table columns may require original PDF inspection.\n'+page['text'],
                'plans','page',f'sbc:{pid}',f'page {page["page"]}',urls.get(pid+'.pdf',''),a['StateCode'],[pid],
                [dict(target='plan:'+pid+':cost',relation='plan_cost_rules',source=id)])
    return records,{p:digest((root/p).read_bytes()) for p in SOURCE_PATHS}


def uploaded_records(files,namespace='plans',year=2025,state='',plan_ids=None):
    if namespace not in {'enrollment','plans'}:raise ValueError('Choose a source namespace for this upload batch')
    records=[];hashes={};warnings=[]
    if sum(len(b) for b in files.values())>100_000_000:raise ValueError('Upload batch exceeds 100 MB')
    for name,data in sorted(files.items()):
        if Path(name).name!=name or '\\' in name:raise ValueError('Upload filename must be a basename')
        suffix=Path(name).suffix.lower();sid='upload:'+digest(data)[:16]
        if sid in hashes:raise ValueError('Duplicate uploaded document content')
        hashes[sid]=digest(data)
        if suffix=='.pdf':
            from pypdf import PdfReader
            reader=PdfReader(io.BytesIO(data));units=[]
            for i,p in enumerate(reader.pages,1):
                text=p.extract_text() or ''
                if not text.strip():warnings.append(f'{name} page {i}: no extracted text; OCR needed');continue
                units.append((f'page {i}',text))
        elif suffix=='.csv':
            rows=csv.DictReader(io.StringIO(data.decode('utf-8-sig')))
            units=[(f'CSV record {i}',json.dumps(row)) for i,row in enumerate(rows,2)]
        elif suffix in {'.txt','.md'}:units=[('document',data.decode('utf-8-sig'))]
        else:raise ValueError('Use PDF, CSV, TXT or MD source documents; not notebooks, archives or answer-key JSON')
        if not units:raise ValueError('No usable text in '+name)
        for i,(locator,text) in enumerate(units):
            id=sid+':'+str(i)
            links=[]
            if i:links=[dict(target=sid+':'+str(i-1),relation='previous_source_page_or_row',source=id)]
            records.append(dict(id=id,text=name+'\n'+text,source_id=sid,filename=name,locator=locator,url='',kind='uploaded',
                                namespace=namespace,year=int(year),state=state,plan_ids=plan_ids or [],links=links))
    return records,hashes,warnings


def chunk_records(records,tokenizer,size=320,overlap=40):
    if not 0<=overlap<size<=450:raise ValueError('Invalid embedding chunk budget')
    docs=[];mapping={}
    for r in records:
        encoded=tokenizer(r['text'],add_special_tokens=False,return_offsets_mapping=True)
        offsets=encoded['offset_mapping'];pieces=[]
        for start in range(0,len(offsets),size-overlap):
            stop=min(start+size,len(offsets));lo,hi=offsets[start][0],offsets[stop-1][1]
            part=dict(r,id=r['id']+'#'+str(len(pieces)),record_id=r['id'],text=r['text'][lo:hi],links=[],token_span=[start,stop])
            pieces.append(part)
            if stop==len(offsets):break
        if not pieces:continue
        docs.extend(pieces);mapping[r['id']]=pieces
    for r in records:
        for d in mapping.get(r['id'],[]):
            for other in mapping[r['id']]:
                if other['id']!=d['id']:d['links'].append(dict(target=other['id'],relation='same_source_record',source=d['id']))
            for edge in r['links']:
                for target in mapping.get(edge['target'],[]):
                    d['links'].append(dict(target=target['id'],relation=edge['relation'],source=d['id']))
    return docs


def save_corpus(path,docs,sources,config,warnings=None):
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    text=''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in docs)
    (path/'documents.jsonl').write_text(text)
    manifest=dict(schema=1,documents_sha256=digest(text.encode()),sources=sources,count=len(docs),
                  embedding_model=config['embedding_model'],embedding_revision=config['embedding_revision'],
                  chunk_tokens=config['chunk_tokens'],chunk_overlap=config['chunk_overlap'],warnings=warnings or [],
                  graph_kind='explicit source joins and structural document links; no inferred entity facts')
    (path/'corpus.json').write_text(json.dumps(manifest,indent=2))
    return path


def load_corpus(path):
    path=Path(path);manifest=json.loads((path/'corpus.json').read_text());data=(path/'documents.jsonl').read_bytes()
    if digest(data)!=manifest['documents_sha256']:raise ValueError('Corpus hash mismatch')
    docs=[json.loads(x) for x in data.decode().splitlines()]
    if len(docs)!=manifest['count']:raise ValueError('Corpus size mismatch')
    return docs,manifest
