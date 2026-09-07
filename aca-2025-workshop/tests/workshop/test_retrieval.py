import unittest
from src.rag.core import BM25, Graph, fuse, allowed, select_context, artifact_export, artifact_import
from pathlib import Path
import tempfile, zipfile


def doc(id,text,namespace='plans',plan='P1',kind='benefit'):
    return dict(id=id,source_id=id,namespace=namespace,year=2025,state='TX',plan_ids=[plan],kind=kind,text=text,locator=id,url='',links=[])


class RetrievalTests(unittest.TestCase):
    def test_bm25_and_no_hits(self):
        index=BM25([doc('a','specialist visit copayment'),doc('b','pregnancy prenatal delivery')])
        self.assertEqual(index.search('prenatal')[0][0],1)
        self.assertEqual(index.search('unicorn'),[])

    def test_filters_fail_closed(self):
        d=doc('a','x')
        self.assertFalse(allowed(d,dict(namespace='enrollment')))
        self.assertFalse(allowed(d,dict(plan_ids=['P2'])))
        self.assertFalse(allowed(d,dict(year=2024)))
        self.assertFalse(allowed(d,dict(state='FL')))
        self.assertTrue(allowed(d,dict(namespace='both',plan_ids=['P1'])))
        with self.assertRaises(ValueError): allowed(d,dict(namespace='plnas'))

    def test_graph_filters_cycles_and_provenance(self):
        docs=[doc('a','delivery'),doc('b','deductible',kind='attributes'),doc('c','other',plan='P2')]
        docs[0]['links']=[dict(target='b',relation='cost_rules',source='a'),dict(target='c',relation='bad',source='a')]
        docs[1]['links']=[dict(target='a',relation='back',source='b')]
        found=Graph(docs).expand([(0,2.)],dict(plan_ids=['P1']),depth=2,limit=5)
        self.assertEqual({i for i,s,p in found},{0,1})
        self.assertTrue(any(p and p[-1]['source']=='a' for i,s,p in found if i==1))

    def test_fusion_deduplicates_and_stable(self):
        self.assertEqual(fuse([[(1,9),(2,8)],[(2,.9)]])[0][0],2)

    def test_budget_refuses_silent_omission(self):
        with self.assertRaises(ValueError): select_context([doc('a','many '*100)],len,20,policy='strict')
        packed,meta=select_context([doc('a','word'),doc('b','x'*100)],len,30,policy='prefix')
        self.assertEqual(len(packed),1)
        self.assertEqual(meta['omitted'],1)

    def test_archive_paths_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); a=root/'a';a.mkdir();(a/'x.json').write_text('{}')
            archive=artifact_export(a,root/'a.zip','corpus')
            loaded=artifact_import(archive,root/'imports','corpus')
            self.assertEqual((loaded/'x.json').read_text(),'{}')
            self.assertEqual(artifact_import(archive,root/'imports','corpus'),loaded)
            bad=root/'bad.zip'
            with zipfile.ZipFile(bad,'w') as z:z.writestr('../escape','x')
            with self.assertRaises(ValueError): artifact_import(bad,root/'imports','corpus')

class ProfileTests(unittest.TestCase):
    def test_exact_tobacco_and_unknown_upload_scope(self):
        a=doc('a','non tobacco rate',kind='rate');a['profile']={'tobacco':False}
        self.assertTrue(allowed(a,{'tobacco':False}))
        self.assertFalse(allowed(a,{'tobacco':True}))
        upload=doc('u','some state benefit',kind='uploaded');upload['state']=''
        self.assertFalse(allowed(upload,{'state':'TX'}))
        upload['plan_ids']=[]
        self.assertFalse(allowed(upload,{'plan_ids':['P1']}))
