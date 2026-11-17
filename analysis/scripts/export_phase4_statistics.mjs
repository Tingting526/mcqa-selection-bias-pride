import fs from 'node:fs/promises';
import path from 'node:path';
import { writeMatrix } from './_csv.mjs';
if (!process.argv[2]) throw new Error('Usage: node export_phase4_statistics.mjs INPUT_DIR [OUTPUT_DIR]');
const input=path.resolve(process.argv[2]);
const out=path.resolve(process.argv[3] ?? input);
const data=JSON.parse(await fs.readFile(path.join(input,'statistics_results.json'),'utf8'));
const records=[];
for(const [name,rows] of [['direction_classification',data.direction_rows],['rq3_accuracy_intervals',data.bootstrap_rows],['rstd_support_diagnostics',data.rstd_support_diagnostics]]) {
  const keys=[...new Set(rows.flatMap(r=>Object.keys(r)))];
  records.push(await writeMatrix(path.join(out,'tables',name+'.csv'),[keys,...rows.map(r=>keys.map(k=>r[k]))]));
}
await fs.writeFile(path.join(out,'csv_export_validation.json'),JSON.stringify({engine:'node:fs RFC4180 CSV',files:records},null,2)+'\n');
console.log(JSON.stringify(records));
