#!/usr/bin/env python3
"""Validate all new numerical cells, claim arithmetic and preserved evidence."""
import difflib
import hashlib
import json
from pathlib import Path
import re
import sys
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[2];OUT=Path((ROOT/'tmp/phase7_active_output.txt').read_text().strip());P6=ROOT/'analysis/outputs/phase6_robustness_20261018';PAPER=ROOT/'Paper'
sys.path.insert(0,str(ROOT/'analysis'))
from src.offline_phase3b import sha256

def main():
    b=json.loads((OUT/'preservation_baseline.json').read_text());backup=Path(b['backup']);unchanged=[]
    for r in b['protected']:
        p=Path(r['path']);assert p.stat().st_size==r['bytes'] and sha256(p)==r['sha256'],p;unchanged.append(r['path'])
    for r in b['paper_before']:
        q=backup/Path(r['path']).relative_to(PAPER);assert sha256(q)==r['sha256']
    oldassets=[]
    for r in b['paper_before']:
        p=Path(r['path']);relative=p.relative_to(PAPER)
        if relative.parts[0] in ['generated','figures'] or relative.as_posix() in ['bibliography.bib','thesis.tex','chapters/introduction.tex','chapters/2related_work.tex']:
            assert sha256(p)==r['sha256'],p;oldassets.append(str(relative))
    d=json.loads((P6/'analysis_results.json').read_text());s=json.loads((P6/'seed_resampling_results.json').read_text());maps=json.loads((OUT/'table_source_map.json').read_text())
    for cell in maps['numeric_cells']:
        file,key=cell['source'].split(':');pool=d[key] if file=='analysis_results.json' else s[key]
        wanted={k:v for k,v in [('model',cell['model']),('benchmark',cell['benchmark']),('source',cell['direction']),('seed',cell['seed']),('group',cell['group']),('grouping',cell['grouping'])] if v is not None}
        matches=[r for r in pool if all(r.get(k)==v for k,v in wanted.items())];assert len(matches)==1,(wanted,len(matches))
        value=matches[0][cell['field']];assert value==cell['value'] and f'{value:.2f}'==cell['display']
    newtext=(PAPER/'generated/robustness_appendix.tex').read_text()+(PAPER/'generated/robustness_transfer_summary.tex').read_text()
    for t in maps['tables']:
        assert '\\label{'+t['label']+'}' in newtext
        for row in t['display_rows']:assert ' & '.join(map(str,row))+r'\\' in newtext
    # Explicit prior coordinates/support cells do not use two-decimal metrics.
    for r in d['equal_budget']:assert ' & '.join(f'{v:.5f}' for v in r['source_prior']) in newtext
    for r in d['mmlu_subjects']:
        if r['model']=='llama-3.1-8b-instruct' and r['grouping']=='individual_subject':
            assert r'\path{'+r['group']+'} & '+str(r['group_population_n'])+' & '+' & '.join(map(str,r['full_default_gold_supports'])) in newtext
    claims=[]
    def claim(name,actual,expected):
        assert actual==expected,(name,actual,expected);claims.append(dict(claim=name,actual=actual,status='pass'))
    p=d['corrected_permutation']
    claim('cyclic RStd decreases, strict unrounded signs',sum(r['delta_cyclic_rstd_pp']<0 for r in p),26)
    claim('consistency increases, exact gained/lost counts',sum(r['consistency_gained_n']>r['consistency_lost_n'] for r in p),24)
    claim('original RStd decreases',sum(r['delta_original_rstd_pp']<0 for r in p),25)
    claim('original accuracy positive/negative/equal from integer transitions',[sum(r['original_correct_gained_n']>r['original_correct_lost_n'] for r in p),sum(r['original_correct_gained_n']<r['original_correct_lost_n'] for r in p),sum(r['original_correct_gained_n']==r['original_correct_lost_n'] for r in p)],[20,6,4])
    expectedcyc={('qwen2.5-7b-instruct','csqa'),('mistral-7b-instruct-v0.3','medqa'),('qwen3-8b','arc'),('qwen3-8b','csqa')};expectedcons={('mistral-7b-instruct-v0.3',ds) for ds in ['mmlu','arc','medqa']}|{('qwen3-8b',ds) for ds in ['arc','csqa','medmcqa']}
    assert {(r['model'],r['benchmark']) for r in p if r['delta_cyclic_rstd_pp']>=0}==expectedcyc
    assert {(r['model'],r['benchmark']) for r in p if r['delta_semantic_consistency_pp']<=0}==expectedcons
    for model,ds,fields in [
        ('mistral-7b-instruct-v0.3','mmlu',dict(delta_original_rstd_pp='-3.60',delta_cyclic_rstd_pp='-0.32',default_semantic_consistency_percent='51.85',pride_semantic_consistency_percent='50.34')),
        ('qwen3-8b','arc',dict(delta_original_accuracy_pp='0.09',default_cyclic_rstd_pp='0.32',pride_cyclic_rstd_pp='1.42',delta_semantic_consistency_pp='-1.26')),
        ('mistral-7b-instruct-v0.3','medqa',dict(delta_original_accuracy_pp='0.25',delta_semantic_consistency_pp='-2.81')),
        ('glm-4-9b-chat-hf','csqa',dict(delta_original_accuracy_pp='3.89',delta_semantic_consistency_pp='8.39'))]:
        r=next(r for r in p if r['model']==model and r['benchmark']==ds)
        for field,value in fields.items():claim(model+' '+ds+' '+field,f'{r[field]:.2f}',value)
    eq=[r for r in d['equal_budget'] if r['source']=='medmcqa'];claim('equal-budget reverse all6 negative',sum(r['transfer_minus_in_domain_accuracy_pp']<0 for r in eq),6)
    claim('equal-budget reverse transfer accuracy change extrema',[f'{min(r["change_vs_primary_transfer_accuracy_pp"] for r in eq):.2f}',f'{max(r["change_vs_primary_transfer_accuracy_pp"] for r in eq):.2f}'],['-0.33','0.25'])
    claim('equal-budget transfer RStd higher forward/reverse',[sum(r['transfer_minus_in_domain_rstd_pp']>0 for r in d['equal_budget'] if r['source']==source) for source in ['medqa','medmcqa']],[5,4])
    for source in ['medqa','medmcqa']:
        rows=[r for r in s['rows'] if r['source']==source]
        counts=[sum(r['transfer_minus_in_domain_accuracy_pp']>1e-9 for r in rows),sum(r['transfer_minus_in_domain_accuracy_pp']< -1e-9 for r in rows),sum(abs(r['transfer_minus_in_domain_accuracy_pp'])<=1e-9 for r in rows)]
        claim(source+' resampling positive/negative/tie',counts,[27,3,0] if source=='medqa' else [2,27,1])
        claim(source+' distinct fitting questions range',[min(r['source_unique_fit_n'] for r in rows),max(r['source_unique_fit_n'] for r in rows)],[36,43] if source=='medqa' else [121,126])
        claim(source+' maximum prior coordinate SD',f'{max(max(r["source_prior_label_sample_sd_probability"]) for r in rows):.4f}','0.0352' if source=='medqa' else '0.0248')
    for model,lo,hi in [('llama-3.1-8b-instruct','-0.13','0.32'),('glm-4-9b-chat-hf','-0.24','0.45')]:
        rows=[r for r in s['rows'] if r['model']==model and r['source']=='medqa'];claim(model+' forward pool contrast range',[f'{min(r["transfer_minus_in_domain_accuracy_pp"] for r in rows):.2f}',f'{max(r["transfer_minus_in_domain_accuracy_pp"] for r in rows):.2f}'],[lo,hi])
    for scheme,n in [('core5',939),('broad8',1604)]:
        rows=[r for r in d['mmlu_subjects'] if r['grouping']==scheme];claim(scheme+' medical sizes',sorted({r['group_population_n'] for r in rows if r['group']=='medical'}),[n])
        m={r['model']:r for r in rows if r['group']=='medical'};rest={r['model']:r for r in rows if r['group']=='remaining_subjects'}
        claim(scheme+' accuracy higher count',sum(m[k]['full_default_original_accuracy_percent']>rest[k]['full_default_original_accuracy_percent'] for k in m),6)
        if scheme=='core5':claim('core consistency higher count',sum(m[k]['full_default_semantic_consistency_percent']>rest[k]['full_default_semantic_consistency_percent'] for k in m),6)
        else:claim('broad Gemma consistency contrast negative',m['gemma-2-9b-it']['full_default_semantic_consistency_percent']<rest['gemma-2-9b-it']['full_default_semantic_consistency_percent'],True)
    # Read the exact active input graph, excluding inactive historical notes.
    active=[]
    def visit(file):
        if file in active:return
        active.append(file);text=file.read_text()
        for name in re.findall(r'\\input\{([^}]+)\}',text):visit(PAPER/(name if name.endswith('.tex') else name+'.tex'))
    visit(PAPER/'thesis.tex');alltex='\n'.join(p.read_text() for p in active)
    labels=re.findall(r'\\label\{([^}]+)\}',alltex);assert len(labels)==len(set(labels))
    refs=re.findall(r'\\(?:ref|eqref|autoref|pageref)\{([^}]+)\}',alltex);assert set(refs)<=set(labels),set(refs)-set(labels)
    keys=set(k.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}',alltex) for k in group.split(','));bib=set(re.findall(r'@\w+\{([^,]+),',(PAPER/'bibliography.bib').read_text()));assert keys<=bib
    oldeq=[];neweq=[]
    for f in backup.rglob('*.tex'):
        rel=f.relative_to(backup)
        if rel.parts[0]=='build':continue
        oldeq.extend(re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}',f.read_text(),re.S));neweq.extend(re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}',(PAPER/rel).read_text(),re.S))
    assert oldeq==neweq and len(oldeq)==21
    results=(PAPER/'chapters/5ResultsAnalysis.tex').read_text();assert results.index(r'\label{sec:results-rq3}')<results.index(r'\subsection{Robustness and Sensitivity Analyses}')<results.index(r'\subsection{Nominal computational workload}')
    assert not re.search(r'No equal-budget or additional-seed|No equal-budget or repeated-split|do not establish that PriDe improves consistency after correction',alltex)
    # All additions are retained as a reviewable diff, including numerical lines.
    diff=[];added=[];newprose=[]
    for pth in sorted(PAPER.rglob('*.tex')):
        if 'build' in pth.parts:continue
        old=backup/pth.relative_to(PAPER);before=old.read_text().splitlines() if old.exists() else [];after=pth.read_text().splitlines()
        lines=list(difflib.unified_diff(before,after,fromfile='before/'+str(pth.relative_to(PAPER)),tofile='after/'+str(pth.relative_to(PAPER)),lineterm=''));diff.extend(lines)
        for line in lines:
            if line.startswith('+') and not line.startswith('+++'):
                if re.search(r'\d',line):added.append(dict(file=str(pth),text=line[1:]))
                if pth.parent.name=='chapters':newprose.append(line[1:])
    (OUT/'manuscript_revision.diff').write_text('\n'.join(diff)+'\n');(OUT/'added_numeric_lines.json').write_text(json.dumps(added,indent=2)+'\n')
    tokens=lambda text:re.findall(r'[a-z]+',text.lower())
    chunks=lambda values:{tuple(values[i:i+12]) for i in range(max(0,len(values)-11))}
    prosechunks=chunks(tokens('\n'.join(newprose)));matches=[]
    for file in (ROOT/'beispiel_arbeit').glob('*.pdf'):
        text=' '.join(p.extract_text() for p in PdfReader(file).pages);overlap=prosechunks&chunks(tokens(text));matches.extend([dict(example=file.name,phrase=' '.join(x)) for x in sorted(overlap)])
    # Generic overlaps, if any, must be reviewed; do not claim none unchecked.
    (OUT/'style_overlap_check.json').write_text(json.dumps(dict(window_tokens=12,matches=matches),indent=2)+'\n')
    log=(PAPER/'build/thesis.log').read_text();warnings=re.findall(r'^.*(?:LaTeX Warning:|Package \S+ Warning:|Overfull \\|Underfull \\|undefined references).*$' ,log,re.M);assert not warnings,warnings
    pdf=PdfReader(PAPER/'build/thesis.pdf');text='\n'.join(page.extract_text() for page in pdf.pages)
    assert '??' not in text and not re.search(r'\b(?:TODO|TBD|PLACEHOLDER|INSERT HERE)\b',text)
    assert len(keys)==16
    report=dict(status='pass',protected_nonmanuscript_files=len(unchanged),all_paper_backups_match=True,original_assets_and_untouched_core_files=oldassets,
        primary_results='All30 primary summaries,12 paired transfers and36 approved intervals unchanged; existing generated numeric tables/three figures byte-identical',numeric_table_cells=len(maps['numeric_cells']),presentation_tables=len(maps['tables']),claim_checks=claims,active_latex_inputs=len(active),active_input_paths=[str(p) for p in active],citation_keys=sorted(keys),labels=len(labels),references=len(refs),equations_unchanged=21,warnings=warnings,pages=len(pdf.pages),pdf_sha256=sha256(PAPER/'build/thesis.pdf'),example_overlap_matches=len(matches),model_dependencies_loaded=any(k.split('.')[0] in ['torch','transformers','vllm'] for k in sys.modules))
    (OUT/'claims_and_structure_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['status','protected_nonmanuscript_files','numeric_table_cells','presentation_tables','active_latex_inputs','pages','example_overlap_matches','pdf_sha256']}))

if __name__=='__main__':main()
