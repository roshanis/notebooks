"""Deterministic retrieval, explicit filters and portable verified artifacts."""
from collections import Counter, deque
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile


def digest(value):
    if not isinstance(value,bytes): value=json.dumps(value,sort_keys=True,ensure_ascii=False).encode()
    return hashlib.sha256(value).hexdigest()


def tokenize(text):
    return re.findall(r'\d{5}[a-z]{2}\d{7}-\d{2}|[a-z0-9]+',text.lower())


class BM25:
    def __init__(self,docs,k1=1.5,b=.75):
        self.counts=[Counter(tokenize(d['text'])) for d in docs]
        self.df=Counter(t for c in self.counts for t in c)
        self.lengths=[sum(c.values()) for c in self.counts]
        self.avg=sum(self.lengths)/max(1,len(docs));self.k1=k1;self.b=b

    def search(self,query,eligible=None,limit=30):
        ids=range(len(self.counts)) if eligible is None else eligible
        terms=set(tokenize(query));n=len(self.counts);scores=[]
        for i in ids:
            c=self.counts[i];score=0.
            for term in terms & c.keys():
                freq=c[term];idf=math.log(1+(n-self.df[term]+.5)/(self.df[term]+.5))
                score+=idf*freq*(self.k1+1)/(freq+self.k1*(1-self.b+self.b*self.lengths[i]/max(self.avg,1)))
            if score>0:scores.append((i,score))
        return sorted(scores,key=lambda x:(-x[1],x[0]))[:limit]


def allowed(d,filters):
    unknown=set(filters)-{'namespace','year','state','plan_ids','source_ids','age','rating_area','date','tobacco'}
    if unknown:raise ValueError('Unsupported filters: '+str(sorted(unknown)))
    ns=filters.get('namespace','both')
    if ns not in {'enrollment','plans','both'}:raise ValueError('Unknown source filter')
    if ns!='both' and d['namespace']!=ns:return False
    if d['year']!=int(filters.get('year',2025)):return False
    if filters.get('state') and d.get('state') not in ('',None,filters['state']):return False
    if filters.get('state') and not d.get('state') and d['kind']=='uploaded':return False
    if d['namespace']=='plans' and filters.get('plan_ids') and not set(d.get('plan_ids',[]))&set(filters['plan_ids']):return False
    if filters.get('source_ids') and d['source_id'] not in filters['source_ids']:return False
    if d['kind']=='rate':
        p=d.get('profile',{})
        for key in ['age','rating_area','tobacco']:
            if filters.get(key) is not None and str(filters[key])!=str(p.get(key)):return False
        date=filters.get('date')
        if date and not p.get('start','')<=date<=p.get('end',''):return False
    return True


class Graph:
    """Source-record graph. Edges represent explicit typed joins, not inferred coverage."""
    def __init__(self,docs):
        self.docs=docs;self.ids={d['id']:i for i,d in enumerate(docs)}
        if len(self.ids)!=len(docs):raise ValueError('Duplicate corpus IDs')
        for d in docs:
            for e in d.get('links',[]):
                if e['target'] not in self.ids or e['source'] not in self.ids:raise ValueError('Unresolvable graph source/target')

    def expand(self,seeds,filters,depth=2,limit=30):
        if not 0<=depth<=3 or not 1<=limit<=200:raise ValueError('Invalid graph traversal budget')
        queue=deque((i,s,[],0) for i,s in seeds if allowed(self.docs[i],filters));best={};visits=0
        while queue and visits<2000:
            i,score,path,hops=queue.popleft();visits+=1
            if i in best and best[i][0]>=score:continue
            best[i]=(score,path)
            if hops==depth:continue
            for e in sorted(self.docs[i].get('links',[]),key=lambda e:(e['relation'],e['target'])):
                j=self.ids[e['target']];d=self.docs[j]
                if not allowed(d,filters) or d['namespace']!=self.docs[i]['namespace']:continue
                # Source provenance does not authorize crossing a plan variant.
                if d['namespace']=='plans' and not set(d['plan_ids'])&set(self.docs[i]['plan_ids']):continue
                if d.get('state') and self.docs[i].get('state') and d['state']!=self.docs[i]['state']:continue
                queue.append((j,score*.8,path+[dict(e,origin=self.docs[i]['id'])],hops+1))
        return [(i,s,p) for i,(s,p) in sorted(best.items(),key=lambda x:(-x[1][0],x[0]))[:limit]]


def fuse(lists,k=60):
    scores=Counter()
    for rows in lists:
        seen=set()
        for rank,(i,_) in enumerate(rows,1):
            if i not in seen:scores[i]+=1/(k+rank);seen.add(i)
    return sorted(scores.items(),key=lambda x:(-x[1],x[0]))


def block(d):return '['+d['id']+'] '+d['text']


def select_context(docs,count_tokens,budget,policy='strict'):
    if policy not in {'strict','prefix','ranked'}:raise ValueError('Unknown context policy')
    selected=[];used=0
    for d in docs:
        size=count_tokens(block(d))
        if used+size>budget:
            if policy=='strict':raise ValueError('Document context exceeds budget. Narrow source filters or explicitly select prefix mode; nothing was silently truncated.')
            if policy=='prefix':break
            continue
        selected.append(d);used+=size
    return selected,dict(policy=policy,selected=len(selected),available=len(docs),omitted=len(docs)-len(selected),evidence_tokens=used)


def artifact_export(directory,output,kind):
    directory=Path(directory);output=Path(output)
    files={str(p.relative_to(directory)):digest(p.read_bytes()) for p in sorted(directory.rglob('*')) if p.is_file()}
    if not files:raise ValueError('Cannot export empty artifact')
    if 'artifact.json' in files:raise ValueError('Reserved artifact filename')
    if any(p.is_symlink() for p in directory.rglob('*')):raise ValueError('Symlinks are not portable artifacts')
    manifest=dict(schema=1,kind=kind,files=files)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as z:
        z.writestr('artifact.json',json.dumps(manifest,sort_keys=True))
        for name in files:z.write(directory/name,name)
    return output


def artifact_import(archive,parent,kind):
    archive=Path(archive);parent=Path(parent)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or len(names)>20000:raise ValueError('Duplicate or excessive archive members')
        if sum(i.file_size for i in z.infolist())>2_000_000_000:raise ValueError('Artifact exceeds size limit')
        for i in z.infolist():
            p=PurePosixPath(i.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in i.filename or stat.S_ISLNK(i.external_attr>>16):raise ValueError('Unsafe archive path')
        if 'artifact.json' not in names:raise ValueError('Missing artifact manifest')
        manifest=json.loads(z.read('artifact.json'))
        if manifest.get('kind')!=kind or manifest.get('schema')!=1:raise ValueError('Wrong artifact kind/schema')
        if set(manifest['files'])!=set(names)-{'artifact.json'}:raise ValueError('Unmanifested artifact contents')
        for name,h in manifest['files'].items():
            if digest(z.read(name))!=h:raise ValueError('Artifact checksum mismatch')
        target=parent/(kind+'-'+digest(manifest)[:16])
        target.mkdir(parents=True,exist_ok=True)
        for name in manifest['files']:
            p=target/name
            if p.exists():
                if not p.is_file() or digest(p.read_bytes())!=manifest['files'][name]:raise ValueError('Existing artifact was modified')
            else:
                p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('xb') as f:f.write(z.read(name))
    return target
