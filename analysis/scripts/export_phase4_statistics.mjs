import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {Workbook} from '@oai/artifact-tool';
const out=process.argv[2];
const data=JSON.parse(await fs.readFile(path.join(out,'statistics_results.json'),'utf8'));
const workbook=Workbook.create();const records=[];let number=0;
const column=n=>n<=26?String.fromCharCode(64+n):String.fromCharCode(64+Math.floor((n-1)/26))+String.fromCharCode(65+(n-1)%26);
for(const [name,rows] of [['direction_classification',data.direction_rows],['rq3_accuracy_intervals',data.bootstrap_rows],['rstd_support_diagnostics',data.rstd_support_diagnostics]]) {
  const keys=[...new Set(rows.flatMap(r=>Object.keys(r)))];
  const convert=v=>v===null||v===undefined?'':typeof v==='object'?JSON.stringify(v):v;
  const matrix=[keys,...rows.map(r=>keys.map(k=>convert(r[k])))];
  const sheet=workbook.worksheets.add('Statistics'+(++number));const range=sheet.getRange(`A1:${column(keys.length)}${matrix.length}`);range.values=matrix;workbook.recalculate();
  const values=range.values;
  for(let i=0;i<matrix.length;i++)for(let j=0;j<keys.length;j++)if(String(matrix[i][j]??'')!==String(values[i][j]??''))throw new Error('Cell conversion mismatch');
  const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';const csv=values.map(r=>r.map(quote).join(',')).join('\r\n')+'\r\n';
  const filename=path.join(out,'tables',name+'.csv');await fs.mkdir(path.dirname(filename),{recursive:true});await fs.writeFile(filename,csv,{flag:'wx'});
  records.push({path:filename,rows:rows.length,sha256:crypto.createHash('sha256').update(csv).digest('hex'),all_cells_verified:true});
}
await fs.writeFile(path.join(out,'csv_export_validation.json'),JSON.stringify({files:records},null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify(records));
