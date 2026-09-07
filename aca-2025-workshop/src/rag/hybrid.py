"""Inspectable hybrid search with hard filters applied to every branch."""
import time
from src.rag.core import BM25,Graph,allowed,fuse
from src.rag.corpus import load_corpus
from src.rag.semantic import load_index


class Hybrid:
    def __init__(self,corpus_path,index_path,embedder):
        self.docs,self.manifest=load_corpus(corpus_path);self.embedder=embedder;self.config=embedder.config
        self.vectors=load_index(index_path,self.manifest,self.config);self.bm25=BM25(self.docs);self.graph=Graph(self.docs)

    def search(self,query,filters=None,method='hybrid',k=None):
        import numpy as np
        if method not in {'bm25','semantic','graph','hybrid'}:raise ValueError('Unknown search method')
        if not isinstance(query,str) or not query.strip():raise ValueError('Enter a question')
        filters=filters or {};eligible=[i for i,d in enumerate(self.docs) if allowed(d,filters)]
        start=time.perf_counter();limit=self.config['candidate_k'];paths={};branches={}
        if not eligible:return dict(results=[],branches={},seconds=0.,graph_seed_method='BM25',eligible=0)
        lex=self.bm25.search(query,eligible,limit)
        branches['bm25']=lex
        sem=[]
        if method in {'semantic','hybrid'}:
            vector=self.embedder.encode([query],query=True)[0];scores=self.vectors[eligible]@vector
            sem=sorted(zip(eligible,map(float,scores)),key=lambda x:(-x[1],x[0]))[:limit];branches['semantic']=sem
        seeds=lex[:3] # Disclosed lexical entry points for graph-only and hybrid ablation.
        expanded=[]
        if method in {'graph','hybrid'}:
            expanded=self.graph.expand(seeds,filters,self.config['graph_depth'],limit)
            paths={i:p for i,s,p in expanded};branches['graph']=[(i,s) for i,s,p in expanded]
        ranked=fuse(list(branches.values()),self.config['rrf_k']) if method=='hybrid' else branches[method]
        rows=[]
        for i,score in ranked[:k or self.config['final_k']]:
            d=self.docs[i]
            if not allowed(d,filters):raise AssertionError('Post-fusion filter violation')
            rows.append(dict(document=d,score=score,graph_path=paths.get(i,[]),
                             ranks={name:next((j for j,(idx,s) in enumerate(items,1) if idx==i),None) for name,items in branches.items()}))
        return dict(results=rows,seconds=time.perf_counter()-start,eligible=len(eligible),graph_seed_method='BM25 top3',
                    branches={name:[self.docs[i]['id'] for i,s in values] for name,values in branches.items()})
