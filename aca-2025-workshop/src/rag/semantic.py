"""Real pinned embeddings, persisted with source and configuration fingerprints."""
import json
from pathlib import Path
from src.rag.core import digest


class Embedder:
    def __init__(self,config,device='cpu',local_path=None):
        from sentence_transformers import SentenceTransformer
        self.config=config
        self.model=SentenceTransformer(local_path or config['embedding_model'],revision=None if local_path else config['embedding_revision'],
                                      device=device,trust_remote_code=False)
        self.model.max_seq_length=512
        self.tokenizer=self.model.tokenizer

    def encode(self,texts,query=False):
        if query:texts=[self.config['embedding_query_prefix']+t for t in texts]
        if any(len(self.tokenizer(t,add_special_tokens=True)['input_ids'])>512 for t in texts):raise ValueError('Embedding input exceeds512 tokens; refusing truncation')
        return self.model.encode(texts,batch_size=32,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=len(texts)>100)


def build_index(corpus_path,output,embedder):
    import numpy as np
    from src.rag.corpus import load_corpus
    docs,manifest=load_corpus(corpus_path)
    if (manifest['embedding_model'],manifest['embedding_revision'])!=(embedder.config['embedding_model'],embedder.config['embedding_revision']):raise ValueError('Chunk tokenizer and embedding revision differ')
    vectors=embedder.encode([d['text'] for d in docs]);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    np.save(output/'vectors.npy',vectors,allow_pickle=False)
    meta=dict(schema=1,corpus_hash=digest(manifest),config=embedder.config,shape=list(vectors.shape),vectors_sha256=digest((output/'vectors.npy').read_bytes()))
    (output/'index.json').write_text(json.dumps(meta,indent=2));return output


def load_index(path,manifest,config):
    import numpy as np
    path=Path(path);meta=json.loads((path/'index.json').read_text())
    if meta['corpus_hash']!=digest(manifest) or meta['config']!=config:raise ValueError('Index does not match this corpus/config; rebuild in a new directory')
    if digest((path/'vectors.npy').read_bytes())!=meta['vectors_sha256']:raise ValueError('Vector checksum mismatch')
    vectors=np.load(path/'vectors.npy',allow_pickle=False)
    if list(vectors.shape)!=meta['shape'] or not np.isfinite(vectors).all():raise ValueError('Malformed vectors')
    return vectors
