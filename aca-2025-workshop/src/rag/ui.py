"""Readable, escaped notebook output with inspectable evidence and graph paths."""
import html,json
from collections import Counter

def show_corpus(docs,manifest):
    from IPython.display import display,HTML
    counts=Counter(d['namespace'] for d in docs)
    rows=''.join(f'<tr><td>{html.escape(ns)}</td><td>{n:,}</td></tr>' for ns,n in counts.items())
    display(HTML('<h3>Your source collection</h3><table><tr><th>Source group</th><th>Chunks</th></tr>'+rows+'</table><p>Source records and extracted pages only. No answer keys are indexed.</p>'))
    for w in manifest.get('warnings',[]):print(w)

def show_search(result):
    from IPython.display import display,HTML
    parts=[f'<p>{len(result["results"])} passages · {result["eligible"]} eligible · {result["seconds"]:.3f}s</p>']
    for row in result['results']:
        d=row['document'];esc=html.escape
        path=' → '.join(e['relation']+' → '+e['target'] for e in row['graph_path']) or 'Direct retrieval seed'
        url=d.get('url','');link=f'<a href="{esc(url,quote=True)}">Original source</a>' if url.startswith('https://') else esc(d.get('filename','Uploaded source'))
        parts.append(f'<details><summary><b>{esc(d["id"])}</b> · {esc(d["locator"])} · score {row["score"]:.4f}</summary><p>{link}</p><p>{esc(path)}</p><pre style="white-space:pre-wrap">{esc(d["text"])}</pre><p>Ranks: {esc(str(row["ranks"]))}</p></details>')
    display(HTML(''.join(parts)))

def show_answer(row):
    from IPython.display import display,HTML
    raw=row.get('raw_output','')
    try:
        p=json.loads(raw);text=p.get('answer',raw);cites=p.get('citations',[])
    except ValueError:text=raw;cites=[]
    display(HTML('<h3>Model answer</h3><p>'+html.escape(text)+'</p><p>Citations: '+html.escape(', '.join(cites))+'</p>'))
    if row.get('evidence',{}).get('omitted'):print('Visible context limit:',row['evidence']['omitted'],'chunks omitted under',row['evidence']['policy'],'policy.')
    print('Automated checks:',row.get('scores'));print('Generation seconds:',row.get('generation_seconds'))

def show_summary(summary):
    from IPython.display import display,HTML
    rows=''
    for r in summary['groups']:
        timing='—' if r['median_generation_seconds'] is None else f'{r["median_generation_seconds"]:.2f}s'
        rows+='<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [r['namespace'],r['condition'],r['attempts'],r['errors'],f'{r["valid_json"]}/{r["successful"]}',f'{r["valid_citation_ids"]}/{r["successful"]}',timing])+'</tr>'
    display(HTML('<h3>Measured execution results</h3><table><tr><th>Dataset</th><th>Condition</th><th>Attempts</th><th>Errors</th><th>Valid JSON / generated</th><th>Valid citation IDs / generated</th><th>Median generation</th></tr>'+rows+'</table><p>Answer accuracy awaits manual grading. Valid citation IDs do not prove entailment.</p>'))

def show_comparison(rows,question_id=None):
    from IPython.display import display,HTML
    if not rows:return
    question_id=question_id or rows[0]['id'];chosen=[r for r in rows if r['id']==question_id]
    columns=[]
    for r in chosen:
        evidence=r.get('evidence',{})
        columns.append('<div style="flex:1;min-width:260px;border:1px solid #ccd6df;border-radius:8px;padding:14px">'
            '<h4>'+html.escape(r['condition'])+'</h4><pre style="white-space:pre-wrap">'+html.escape(r['error'] or r['raw_output'])+'</pre>'
            '<details><summary>Evidence and prompt</summary><pre style="white-space:pre-wrap">'+html.escape(json.dumps(r.get('prompt'),indent=2))+'</pre></details>'
            '<p>Input tokens: '+str(evidence.get('input_tokens','unavailable'))+' · Omitted chunks: '+str(evidence.get('omitted','unavailable'))+'</p></div>')
    display(HTML('<h3>'+html.escape(chosen[0]['question'])+'</h3><div style="display:flex;gap:12px;flex-wrap:wrap">'+''.join(columns)+'</div>'))
