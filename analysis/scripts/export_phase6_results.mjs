import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {Workbook} from '@oai/artifact-tool';
const out=process.argv[2];
const d=JSON.parse(await fs.readFile(path.join(out,'analysis_results.json'),'utf8'));
const control=JSON.parse(await fs.readFile(path.join(out,'seed_control_results.json'),'utf8'));
const resampling=JSON.parse(await fs.readFile(path.join(out,'seed_resampling_results.json'),'utf8'));
const datasets=[['equal_budget_results',d.equal_budget],['calibration_seed_sensitivity',[...resampling.rows,...control.rows]],['corrected_permutation_robustness',d.corrected_permutation],['mmlu_medical_subject_analysis',d.mmlu_subjects]];
const column=n=>{let s='';while(n){n--;s=String.fromCharCode(65+n%26)+s;n=Math.floor(n/26);}return s;};
const records=[];const workbook=Workbook.create();let index=0;
for(const [name,rows] of datasets){
  const keys=[...new Set(rows.flatMap(r=>Object.keys(r)))];
  const convert=v=>v===null||v===undefined?'NA':typeof v==='object'?JSON.stringify(v):v;
  const matrix=[keys,...rows.map(r=>keys.map(k=>convert(r[k])))];
  const sheet=workbook.worksheets.add('Analysis'+(++index));const range=sheet.getRange(`A1:${column(keys.length)}${matrix.length}`);range.values=matrix;
  const values=range.values;
  for(let i=0;i<matrix.length;i++)for(let j=0;j<keys.length;j++)if(String(matrix[i][j])!==String(values[i][j]))throw new Error(`Cell mismatch ${name} ${i} ${j}`);
  const quote=v=>'"'+String(v).replaceAll('"','""')+'"';const csv=values.map(r=>r.map(quote).join(',')).join('\r\n')+'\r\n';
  const file=path.join(out,name+'.csv');
  try { const existing=await fs.readFile(file,'utf8');if(existing!==csv)throw new Error('Existing CSV differs from verified rebuild'); }
  catch(error){if(error.code!=='ENOENT')throw error;await fs.writeFile(file,csv,{flag:'wx'});}
  records.push({path:file,rows:rows.length,columns:keys.length,sha256:crypto.createHash('sha256').update(csv).digest('hex'),all_cells_verified:true});
}
workbook.recalculate();
const summary=workbook.worksheets.add('Preview');
summary.getRange('A1:E1').values=[['Model','Target','Source n','Transfer − In-domain (pp)','Transfer RStd (pp)']];
summary.getRange('A2:E13').values=d.equal_budget.map(r=>[r.model.replace('-instruct','').replace('-chat-hf',''),r.target,r.source_calibration_n,r.transfer_minus_in_domain_accuracy_pp,r.transfer_rstd_pp]);
summary.getRange('A1:E13').format.font={name:'Arial',size:10};summary.getRange('A1:E1').format.fill='#E6ECF2';summary.getRange('A1:E1').format.font.bold=true;
summary.getRange('A:A').format.columnWidth=25;summary.getRange('B:B').format.columnWidth=13;summary.getRange('C:C').format.columnWidth=12;summary.getRange('D:D').format.columnWidth=34;summary.getRange('E:E').format.columnWidth=25;
summary.getRange('A1:E13').format.rowHeight=23;summary.getRange('D2:E13').setNumberFormat('0.00');workbook.recalculate();
const inspection=await workbook.inspect({kind:'table',range:'Preview!A1:E4',include:'values',tableMaxRows:4,tableMaxCols:5});
await fs.writeFile(path.join(out,'tables','artifact_preview_inspection.txt'),inspection.ndjson);
const image=await workbook.render({sheetName:'Preview',range:'A1:E13',scale:1.5});
await fs.writeFile(path.join(out,'tables','equal_budget_preview.png'),new Uint8Array(await image.arrayBuffer()));
await fs.writeFile(path.join(out,'csv_export_validation.json'),JSON.stringify({engine:'@oai/artifact-tool',files:records,missing_data:'NA for undefined/not-applicable; arrays encoded as JSON; units in columns and units field'},null,2)+'\n');
console.log(JSON.stringify(records));
