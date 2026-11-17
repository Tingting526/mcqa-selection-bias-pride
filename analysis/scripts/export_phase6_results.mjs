import fs from 'node:fs/promises';
import path from 'node:path';
import { writeMatrix } from './_csv.mjs';
if (!process.argv[2]) throw new Error('Usage: node export_phase6_results.mjs INPUT_DIR [OUTPUT_DIR]');
const input=path.resolve(process.argv[2]);
const out=path.resolve(process.argv[3] ?? input);
const d=JSON.parse(await fs.readFile(path.join(input,'analysis_results.json'),'utf8'));
const control=JSON.parse(await fs.readFile(path.join(input,'seed_control_results.json'),'utf8'));
const resampling=JSON.parse(await fs.readFile(path.join(input,'seed_resampling_results.json'),'utf8'));
const datasets=[['equal_budget_results',d.equal_budget],['calibration_seed_sensitivity',[...resampling.rows,...control.rows]],['corrected_permutation_robustness',d.corrected_permutation],['mmlu_medical_subject_analysis',d.mmlu_subjects]];
const records=[];
for(const [name,rows] of datasets){
  const keys=[...new Set(rows.flatMap(r=>Object.keys(r)))];
  const matrix=[keys,...rows.map(r=>keys.map(k=>r[k]))];
  records.push(await writeMatrix(path.join(out,name+'.csv'),matrix,'NA'));
}
await fs.writeFile(path.join(out,'csv_export_validation.json'),JSON.stringify({engine:'node:fs RFC4180 CSV',files:records,missing_data:'NA for undefined/not-applicable; arrays encoded as JSON; units in columns and units field'},null,2)+'\n');
console.log(JSON.stringify(records));
