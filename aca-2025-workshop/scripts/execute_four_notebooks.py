"""Execute four workshop notebooks in fresh kernels; enable documented smoke controls."""
import argparse,json,sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);handoff={};report=[]
    for path in sorted(args.input.glob('*.ipynb')):
        nb=nbformat.read(path,as_version=4);number=path.name[:2];changes=[]
        for cell in nb.cells:
            if cell.cell_type!='code':continue
            for old,new in [('INSTALL_DEPENDENCIES = True','INSTALL_DEPENDENCIES = False'),('RUN_LLM = False','RUN_LLM = True'),
                            ('RUN_TRAINING = False','RUN_TRAINING = True'),('RUN_COMPARISON = False','RUN_COMPARISON = True'),
                            ('DEVICE = "auto"','DEVICE = "cpu"')]:
                if old in cell.source:cell.source=cell.source.replace(old,new);changes.append(old+' -> '+new)
            if 'config = json.loads' in cell.source:
                cell.source+='\nimport torch\ntorch.set_num_threads(6)\n'
            if number=='04':
                cell.source=cell.source.replace('USE_ADAPTER = False','USE_ADAPTER = True').replace('QUESTIONS_PER_DATASET = 2','QUESTIONS_PER_DATASET = 1').replace('DOCUMENT_POLICY = "strict"','DOCUMENT_POLICY = "prefix"')
                cell.source=cell.source.replace('ADAPTER = upload_artifact("adapter", OUTPUTS) if USE_ADAPTER else None',
                    'from src.rag.core import artifact_import\nADAPTER = artifact_import('+repr(handoff['adapter_zip'])+', OUTPUTS / "imports", "adapter")')
            if number in {'02','03','04'}:
                cell.source=cell.source.replace('CORPUS = ROOT / "sample" if CORPUS_SOURCE == "Sample" else upload_artifact("corpus", OUTPUTS)',
                    'from src.rag.core import artifact_import\nCORPUS = artifact_import('+repr(handoff['corpus_zip'])+', OUTPUTS / "imports", "corpus")')
            if number=='04':
                cell.source=cell.source.replace('INDEX_SOURCE = "Build"','INDEX_SOURCE = "Upload index artifact"').replace('INDEX = upload_artifact("index", OUTPUTS)',
                    'INDEX = artifact_import('+repr(handoff['index_zip'])+', OUTPUTS / "imports", "index")')
        state=args.output/(path.stem+'-state.json')
        nb.cells.append(nbformat.v4.new_code_cell('Path('+repr(str(state.resolve()))+').write_text(json.dumps({k: str(globals()[k]) for k in ["corpus_zip","index_zip","adapter_zip","results_zip"] if k in globals()}))'))
        kernel=KernelManager(kernel_name='python3');kernel.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
        try:
            NotebookClient(nb,km=kernel,timeout=600,resources={'metadata':{'path':str(args.output.resolve())}}).execute()
        finally:
            if kernel.has_kernel:kernel.shutdown_kernel(now=False)
        handoff.update(json.loads(state.read_text()))
        nb.metadata['workshop'].update(executed=True,validation_overrides=changes,smoke_only=True)
        nbformat.validate(nb)
        with (args.output/path.name).open('x') as f:nbformat.write(nb,f)
        record=dict(notebook=path.name,code_cells=sum(c.cell_type=='code' for c in nb.cells)-1,errors=0,overrides=changes)
        report.append(record);print(json.dumps(record),flush=True)
        (args.output/'execution.json').write_text(json.dumps(dict(notebooks=report,handoff=handoff),indent=2))
if __name__=='__main__':main()
