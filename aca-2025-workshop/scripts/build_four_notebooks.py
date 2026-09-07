"""Generate four independent Colab notebooks with identical embedded runtime and sample corpus."""
import argparse,base64,hashlib,io,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    import nbformat as n
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    shared=[*sorted((ROOT/'src').glob('*.py')),*sorted((ROOT/'src/rag').glob('*.py')),ROOT/'config/rag.json',ROOT/'requirements-rag.txt',
            ROOT/'data/training_examples.jsonl',ROOT/'data/validation_examples.jsonl',ROOT/'eval/questions.jsonl',ROOT/'eval/split_manifest.json',
            *[ROOT/'eval/plans_2025'/name for name in ['training.jsonl','validation.jsonl','questions.jsonl','split_manifest.json']]]
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        snapshot=ROOT/'data/workshop_snapshot'
        snapshot_meta=json.loads((snapshot/'snapshot.json').read_text())
        for path in shared:
            relative=str(path.relative_to(ROOT))
            if relative in snapshot_meta['files']:
                path=snapshot/relative
                if hashlib.sha256(path.read_bytes()).hexdigest()!=snapshot_meta['files'][relative]:raise ValueError('Workshop dataset snapshot changed')
            z.write(path,relative)
        z.write(snapshot/'snapshot.json','workshop_dataset.json')
        for name in ['documents.jsonl','corpus.json']:z.write(a.corpus/name,'sample/'+name)
    encoded=base64.b64encode(buffer.getvalue()).decode();sha=hashlib.sha256(buffer.getvalue()).hexdigest()
    bootstrap=f'''#@title 1. Load the included workshop code and sample sources
import base64, hashlib, io, json, sys, tempfile, zipfile
from pathlib import Path
payload = base64.b64decode("{encoded}")
assert hashlib.sha256(payload).hexdigest() == "{sha}"
ROOT = Path(tempfile.gettempdir()) / "aca-workshop-{sha[:12]}"
ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as z:
    for name in z.namelist():
        target = ROOT / name
        if not target.resolve().is_relative_to(ROOT.resolve()): raise ValueError("Unsafe package path")
        data = z.read(name)
        if target.exists():
            if target.read_bytes() != data: raise ValueError("Existing workshop code changed; use a fresh runtime")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as f: f.write(data)
sys.path.insert(0, str(ROOT))
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
config = json.loads((ROOT / "config/rag.json").read_text())
print("Workshop source snapshot loaded. Nothing has been trained or generated yet.")'''
    install='''#@title 2. Install the pinned notebook dependencies
INSTALL_DEPENDENCIES = True #@param {type:"boolean"}
print((ROOT / "requirements-rag.txt").read_text())
if INSTALL_DEPENDENCIES:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements-rag.txt")], check=True)
from src.rag.notebook import unique, upload_artifact, export_artifact, download
from src.rag.corpus import load_corpus
from src.rag.core import digest
from src.rag.ui import show_corpus, show_search, show_answer, show_summary, show_comparison
print("Dependencies ready. Use a GPU for extended generation/training; retrieval can run on CPU.")'''
    corpus='''#@title Choose the corpus saved by notebook 01, or the identical included sample
CORPUS_SOURCE = "Sample" #@param ["Sample", "Upload corpus artifact"]
CORPUS = ROOT / "sample" if CORPUS_SOURCE == "Sample" else upload_artifact("corpus", OUTPUTS)
docs, manifest = load_corpus(CORPUS)
show_corpus(docs, manifest)'''
    filters='''#@title Choose the question and source filters
QUESTION = "How many plan selections did Kentucky have in 2025?" #@param {type:"string"}
SOURCE_FILTER = "enrollment" #@param ["enrollment", "plans", "both"]
STATE = "KY" #@param {type:"string"}
PLAN_IDS = "" #@param {type:"string"}
SOURCE_IDS = "state_csv" #@param {type:"string"}
AGE = 0 #@param {type:"integer"}
RATING_AREA = "" #@param {type:"string"}
TOBACCO = "unspecified" #@param ["unspecified", "yes", "no"]
RATE_DATE = "" #@param {type:"string"}
if "data/raw/state.csv" not in manifest["sources"] and STATE == "KY" and SOURCE_IDS == "state_csv":
    namespaces = {d["namespace"] for d in docs}
    SOURCE_FILTER = next(iter(namespaces)) if len(namespaces) == 1 else "both"
    STATE, SOURCE_IDS, PLAN_IDS = "", "", ""
    print("Using your uploaded collection; sample Kentucky filters were cleared.")
filters = {"namespace": SOURCE_FILTER, "year": 2025}
if STATE.strip(): filters["state"] = STATE.strip().upper()
if PLAN_IDS.strip(): filters["plan_ids"] = [x.strip() for x in PLAN_IDS.split(",")]
if SOURCE_IDS.strip(): filters["source_ids"] = [x.strip() for x in SOURCE_IDS.split(",")]
if AGE: filters["age"] = AGE
if RATING_AREA.strip(): filters["rating_area"] = RATING_AREA.strip()
if TOBACCO != "unspecified": filters["tobacco"] = TOBACCO == "yes"
if RATE_DATE.strip(): filters["date"] = RATE_DATE.strip()
question = {"id": "interactive", "question": QUESTION, "filters": filters}
print("Filters apply before retrieval and graph traversal:", filters)'''
    index='''#@title Build or import the BM25 + semantic + graph index
INDEX_SOURCE = "Build" #@param ["Build", "Upload index artifact"]
from src.rag.semantic import Embedder, build_index
from src.rag.hybrid import Hybrid
embedder = Embedder(config)
if INDEX_SOURCE == "Build":
    INDEX = build_index(CORPUS, unique(OUTPUTS, "index"), embedder)
else:
    INDEX = upload_artifact("index", OUTPUTS)
rag = Hybrid(CORPUS, INDEX, embedder)
print("Index verified against this exact corpus and embedding configuration.")'''
    md,code=n.v4.new_markdown_cell,n.v4.new_code_cell
    def base(title,description):return [md('# '+title+'\n\n'+description+'\n\n**Historical 2025 sources. No fine-tuning or paid API runs automatically.** Every notebook includes the same sample so you can open it independently. For your own documents, transfer the exported ZIP artifacts between notebooks.'),code(bootstrap,metadata={'cellView':'form'}),code(install)]
    one=base('01 · Documents + default LLM','Upload documents, inspect the extracted text, and give selected document context directly to the unchanged LLM. There is no query-based retrieval in this baseline.')
    one += [md('### Load sample documents or upload your own\nPDF, CSV, TXT and MD are supported. Upload source documents only; do not upload answer keys. Image-only PDF pages need OCR and are reported. Upload metadata applies to the whole batch.'),code('''DATA_SOURCE = "Sample" #@param ["Sample", "Upload documents"]
UPLOAD_NAMESPACE = "plans" #@param ["enrollment", "plans"]
UPLOAD_STATE = "" #@param {type:"string"}
UPLOAD_PLAN_IDS = "" #@param {type:"string"}
if DATA_SOURCE == "Sample":
    CORPUS = ROOT / "sample"
else:
    from google.colab import files
    from src.rag.corpus import uploaded_records, chunk_records, save_corpus
    from src.rag.semantic import Embedder
    uploaded = files.upload()
    records, hashes, warnings = uploaded_records(uploaded, UPLOAD_NAMESPACE, state=UPLOAD_STATE.upper(),
        plan_ids=[s.strip() for s in UPLOAD_PLAN_IDS.split(",") if s.strip()])
    embedder = Embedder(config)
    chunks = chunk_records(records, embedder.tokenizer, config["chunk_tokens"], config["chunk_overlap"])
    CORPUS = save_corpus(unique(OUTPUTS, "corpus"), chunks, hashes, config, warnings)
docs, manifest = load_corpus(CORPUS)
show_corpus(docs, manifest)
print("Available source IDs (first 20):", sorted({d["source_id"] for d in docs})[:20])'''),code(filters),md('### Inspect the selected documents\nClear the sample KY/state_csv filters when using your own documents. The baseline uses stable source order, not relevance ranking.'),code('''from src.rag.core import allowed
selected = sorted([d for d in docs if allowed(d, filters)], key=lambda d: d["id"])
print(len(selected), "selected chunks")
for d in selected[:3]:
    print(d["id"], d["locator"], "\\n", d["text"][:800], "\\n")'''),md('### Run the unchanged LLM\n**Default: partial-document baseline.** Prefix mode reports what was omitted. Strict mode stops if the selected documents exceed the prompt budget. Prefix mode is an explicit partial-document baseline: it takes complete chunks in stable order and reports omitted chunks. It is not full-document coverage.'),code('''RUN_LLM = False #@param {type:"boolean"}
DEVICE = "auto" #@param ["auto", "cpu", "cuda"]
DOCUMENT_POLICY = "prefix" #@param ["prefix", "strict"]
if RUN_LLM:
    from src.rag.experiment import load_llm, answer
    model, tokenizer = load_llm(config, DEVICE)
    result = answer(question, docs, tokenizer, config, model, mode="documents", policy=DOCUMENT_POLICY)
    show_answer(result)
    result_dir = unique(OUTPUTS, "baseline"); result_dir.mkdir()
    (result_dir / "prediction.json").write_text(json.dumps(result, indent=2))
    baseline_zip = export_artifact(result_dir, "baseline", OUTPUTS)
else:
    print("Model execution is off. Enable RUN_LLM to generate an actual baseline answer.")'''),md('### Save the corpus for notebook 02\nThe corpus ZIP contains source text and metadata. Keep it; Colab runtime storage is temporary.'),code('''corpus_zip = export_artifact(CORPUS, "corpus", OUTPUTS)
DOWNLOAD_CORPUS = False #@param {type:"boolean"}
if DOWNLOAD_CORPUS: download(corpus_zip)''')]
    two=base('02 · BM25 + semantic + graph RAG','Build the three retrieval components, inspect what each retrieves, and generate a source-cited answer. Semantic vectors are real BGE embeddings. The graph uses explicit source joins and document-structure links; graph-only retrieval uses BM25 seed passages.')
    two += [code(corpus),code(index),code(filters),md('### Compare retrieval methods\nScores are ranking signals, not confidence probabilities. Graph paths identify why a related passage was included. For uploaded documents, structural links do not imply clinical or coverage relationships.'),code('''for method in ["bm25", "semantic", "graph", "hybrid"]:
    print("\n", method.upper())
    show_search(rag.search(QUESTION, filters, method=method))'''.replace('print("\n",','print("\\n",')),md('### Generate a grounded answer'),code('''RUN_LLM = False #@param {type:"boolean"}
DEVICE = "auto" #@param ["auto", "cpu", "cuda"]
if RUN_LLM:
    from src.rag.experiment import load_llm, answer
    model, tokenizer = load_llm(config, DEVICE)
    result = answer(question, docs, tokenizer, config, model, rag, "rag")
    show_answer(result)
    result_dir = unique(OUTPUTS, "rag-answer"); result_dir.mkdir()
    (result_dir / "prediction.json").write_text(json.dumps(result, indent=2))
    export_artifact(result_dir, "rag-answer", OUTPUTS)
else:
    print("Retrieval is real; LLM generation is off until RUN_LLM is enabled.")'''),md('### Export the index\nNotebook 04 needs both this index artifact and the matching corpus artifact.'),code('''index_zip = export_artifact(INDEX, "index", OUTPUTS)
DOWNLOAD_INDEX = False #@param {type:"boolean"}
if DOWNLOAD_INDEX: download(index_zip)''')]
    three=base('03 · Fine-tune the LLM','Train a LoRA adapter on reviewed prompt/answer examples and save it for notebook 04. Documents alone are not supervised training examples. Start from a fresh base model, and reserve final test questions for comparison.')
    three += [code(corpus),md('### Load reviewed examples\nThe included examples train evidence use across both ACA source groups. They are separate from the retrieval corpus. For your own documents, upload one JSON file with `train`, `validation`, and `test` arrays; each training/validation row needs id, question and chat messages ending with the target assistant answer. Test rows need id, question and filters. Keep questions disjoint.'),code('''TRAINING_SOURCE = "Bundled ACA" #@param ["Bundled ACA", "Upload reviewed examples"]
from src.rag.experiment import training_rows, validate_training
if TRAINING_SOURCE == "Bundled ACA":
    if manifest["sources"].get("data/raw/state.csv") is None:
        raise ValueError("For uploaded documents, supply reviewed examples for that corpus.")
    train_rows, validation_rows, test_rows = training_rows(ROOT)
else:
    from google.colab import files
    uploaded = files.upload()
    if len(uploaded) != 1: raise ValueError("Upload one reviewed training JSON file")
    dataset = json.loads(next(iter(uploaded.values())))
    train_rows, validation_rows, test_rows = [dataset[k] for k in ["train", "validation", "test"]]
    validate_training(train_rows, validation_rows, test_rows)
print("Train:", len(train_rows), "Validation:", len(validation_rows), "Held-out test:", len(test_rows))
print("Training question:", train_rows[0]["question"])
print("No final test answers are displayed or indexed here.")'''),md('### Train and reload the saved adapter\nUse an NVIDIA GPU for a full run. Two-step smoke mode verifies mechanics only and must not be presented as a fine-tuned-quality result. Download the adapter before the session ends.'),code('''RUN_TRAINING = False #@param {type:"boolean"}
DEVICE = "auto" #@param ["auto", "cpu", "cuda"]
SMOKE_ONLY = True #@param {type:"boolean"}
if RUN_TRAINING:
    import gc, torch
    from src.rag.experiment import load_llm, train
    model, tokenizer = load_llm(config, DEVICE)
    model, ADAPTER = train(model, tokenizer, config, train_rows[:2] if SMOKE_ONLY else train_rows,
        validation_rows[:1] if SMOKE_ONLY else validation_rows, unique(OUTPUTS, "adapter"), digest(manifest),
        steps=2 if SMOKE_ONLY else None)
    (ADAPTER / "dataset.json").write_text(json.dumps({"train": train_rows, "validation": validation_rows, "test": test_rows}))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    model, tokenizer = load_llm(config, DEVICE, ADAPTER, digest(manifest))
    print("Saved adapter reloaded successfully. Smoke only:", SMOKE_ONLY)
    adapter_zip = export_artifact(ADAPTER, "adapter", OUTPUTS)
else:
    print("Training is off. Set RUN_TRAINING=True; set SMOKE_ONLY=False for the complete training set.")'''),code('''DOWNLOAD_ADAPTER = False #@param {type:"boolean"}
if DOWNLOAD_ADAPTER:
    if not RUN_TRAINING: raise ValueError("Train the adapter first")
    download(adapter_zip)''')]
    four=base('04 · Compare the four conditions','Run the same questions through base+documents, base+RAG, adapter+documents and adapter+RAG. The model, generation budget and prompt template are shared. Base/adapter prompts must match for the same evidence mode. No condition gets extra numeric tools.')
    four += [code(corpus),code(index),md('### Load the adapter and choose validation or final test\nUse validation while changing retrieval or training settings. Switch to test only after freezing choices. Without an adapter, the two adapter conditions are explicitly marked unrun.'),code('''USE_ADAPTER = False #@param {type:"boolean"}
ADAPTER = upload_artifact("adapter", OUTPUTS) if USE_ADAPTER else None
SPLIT = "validation" #@param ["validation", "test"]
QUESTIONS_PER_DATASET = 2 #@param {type:"integer"}
from src.rag.experiment import training_rows
from src.rag.evaluate import questions_from_rows
if ADAPTER:
    dataset = json.loads((ADAPTER / "dataset.json").read_text())
    rows = dataset[SPLIT]
    print("Adapter smoke-only:", json.loads((ADAPTER / "training.json").read_text())["smoke_only"])
elif manifest["sources"].get("data/raw/state.csv"):
    _, validation_rows, test_rows = training_rows(ROOT)
    rows = validation_rows if SPLIT == "validation" else test_rows
else:
    from google.colab import files
    uploaded = files.upload()
    if len(uploaded) != 1: raise ValueError("Upload one evaluation JSONL file")
    rows = [json.loads(line) for line in next(iter(uploaded.values())).decode().splitlines() if line.strip()]
if rows and "namespace" in rows[0] and "filters" not in rows[0]:
    rows = questions_from_rows(rows)
questions = []
for ns in ["enrollment", "plans"]:
    questions += [r for r in rows if r.get("namespace", r.get("filters",{}).get("namespace")) == ns][:QUESTIONS_PER_DATASET]
if not questions: raise ValueError("Evaluation questions need namespace or filters.namespace")
print("Selected", len(questions), SPLIT, "questions. No reference answers are passed to the models.")'''),md('### Check retrieval separately\nThese source-recall scores use existing citation sets as partial judgments. They are useful diagnostics, not a complete measure of relevance or answer accuracy.'),code('''RUN_RETRIEVAL_EVAL = True #@param {type:"boolean"}
if RUN_RETRIEVAL_EVAL:
    from src.rag.evaluate import retrieval_ablation
    retrieval_report = retrieval_ablation(rag, questions, unique(OUTPUTS, "retrieval-eval"))
    for group in retrieval_report["summary"]: print(group)'''),md('### Run the paired model comparison\n**Strict** document mode can fail when all selected text exceeds the budget; those failures remain visible. **Prefix** is an explicit partial-document baseline with a reported omission count. Neither mode does query-based document ranking. Adapter conditions use exactly the same document or RAG prompt as their base counterparts.'),code('''RUN_COMPARISON = False #@param {type:"boolean"}
DEVICE = "auto" #@param ["auto", "cpu", "cuda"]
DOCUMENT_POLICY = "prefix" #@param ["prefix", "strict"]
if RUN_COMPARISON:
    from src.rag.experiment import comparison
    RUN = comparison(questions, CORPUS, INDEX, embedder, config, unique(OUTPUTS, "comparison"),
                     adapter=ADAPTER, device=DEVICE, policy=DOCUMENT_POLICY)
    show_summary(json.loads((RUN / "summary.json").read_text()))
    comparison_rows = [json.loads(line) for line in (RUN / "predictions.jsonl").read_text().splitlines()]
    show_comparison(comparison_rows)
    results_zip = export_artifact(RUN, "comparison", OUTPUTS)
else:
    print("Comparison not run. Enable RUN_COMPARISON when the selected artifacts and split are ready.")'''),md('### Review the answers before declaring a winner\nThe results artifact contains exact prompts, retrieved passages, graph paths, timings, error rows, a shuffled manual-review worksheet and a separate condition key. Grade correctness, citation support and preserved caveats. Review enrollment and plan benefits separately.'),code('''DOWNLOAD_RESULTS = False #@param {type:"boolean"}
if DOWNLOAD_RESULTS:
    if not RUN_COMPARISON: raise ValueError("Run the comparison first")
    download(results_zip)''')]
    for name,cells in [('01_documents_default_llm',one),('02_hybrid_rag',two),('03_fine_tune_llm',three),('04_compare_all',four)]:
        nb=n.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python3',language='python'),
                              colab=dict(name=name+'.ipynb'),workshop=dict(runtime_sha256=sha,executed=False)))
        n.validate(nb)
        for cell in nb.cells:
            if cell.cell_type=='code':compile(cell.source,name,'exec')
        with (a.output/(name+'.ipynb')).open('x') as f:n.write(nb,f)
        print(name,len(cells),'cells')
if __name__=='__main__':main()
