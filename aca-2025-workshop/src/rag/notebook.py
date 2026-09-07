"""Notebook conveniences. Uploads require explicit calls and artifact kinds."""
import json
from pathlib import Path
from uuid import uuid4
from src.rag.core import artifact_import,artifact_export


def unique(root,label):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    return root/(label+'-'+uuid4().hex[:10])


def upload_artifact(kind,root):
    from google.colab import files
    uploaded=files.upload()
    if len(uploaded)!=1:raise ValueError('Upload exactly one '+kind+' ZIP')
    name,data=next(iter(uploaded.items()))
    if not name.lower().endswith('.zip'):raise ValueError('Expected a ZIP artifact')
    path=unique(root,kind).with_suffix('.zip')
    with path.open('xb') as f:f.write(data)
    return artifact_import(path,Path(root)/'imports',kind)


def export_artifact(path,kind,root):
    out=artifact_export(path,unique(root,kind).with_suffix('.zip'),kind)
    from IPython.display import display,FileLink
    display(FileLink(str(out)));return out


def download(path):
    try:
        from google.colab import files
        files.download(str(path))
    except ImportError:print('Saved:',path)
