// Export saved scientific JSON to CSV; no inference or refitting.
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { writeMatrix, parseCSV } from './_csv.mjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
if (!process.argv[2]) throw new Error('Usage: node export_phase3b_tables.mjs INPUT_DIR [OUTPUT_DIR] [--update-ledger]');
const input=path.resolve(process.argv[2]);
const output=path.resolve(process.argv[3] && !process.argv[3].startsWith('--') ? process.argv[3] : input);
const updateLedger=process.argv.includes('--update-ledger');
const results=JSON.parse(await fs.readFile(path.join(input,'analysis_results.json'),'utf8'));
const sha=text=>crypto.createHash('sha256').update(text).digest('hex');
const quote=value=>'"'+String(value??'').replaceAll('"','""')+'"';
const cell=value=>value===null||value===undefined?'':typeof value==='object'?JSON.stringify(value):value;
const manifest=[];
async function exportMatrix(filename,matrix) {
  manifest.push(await writeMatrix(filename,matrix));
}
async function exportRows(relative,rows) {
  if(!rows.length) return; // Missing empirical results are not fabricated.
  const columns=[...new Set(rows.flatMap(row=>Object.keys(row)))];
  await exportMatrix(path.join(output,relative),[columns,...rows.map(row=>columns.map(key=>cell(row[key])))]);
}

await exportRows('reproduction_validation.csv',results.validation);
await exportRows('tables/rq1_reproduction.csv',results.summary_rows.filter(row=>row.rq==='RQ1'));
await exportRows('tables/rq2_medical.csv',results.summary_rows.filter(row=>row.rq==='RQ2'));
await exportRows('tables/rq2_five_benchmark_comparison.csv',results.summary_rows);
await exportRows('tables/label_metrics.csv',results.label_rows);
await exportRows('tables/calibration_draws.csv',results.calibration_draws);
await exportRows('tables/metric_availability.csv',results.metric_availability);
await exportRows('tables/nominal_workload.csv',results.nominal_workload);
await exportRows('tables/rq3_availability.csv',results.transfer_availability);
if(results.permutation_metrics.length) await exportRows('tables/verified_permutation_metrics.csv',results.permutation_metrics);
if(results.transfer_results.length) {
  const rows=[],contrasts=[],priors=[];
  for(const result of results.transfer_results) for(const [method,metrics] of Object.entries(result.metrics)) {
    rows.push({model:result.model,source:result.source,target:result.target,method,protocol:result.signature.protocol,
      n:metrics.n,accuracy_percent:metrics.accuracy,rstd_pp:metrics.rstd,recalls_percent:metrics.recalls,supports:metrics.supports,
      source_estimation_n:result.source_partition.estimation.length,target_estimation_n:result.target_partition.estimation.length,
      target_heldout_ids_sha256:sha(JSON.stringify(result.target_partition.heldout)),source_prior:result.source_prior,target_prior:result.target_prior,
      contrasts_pp:result.contrasts,uncertainty:'Point estimates; no CI',version:result.version,
      source_manifest:result.provenance.source.manifest,target_manifest:result.provenance.target.manifest});
  }
  await exportRows('tables/rq3_results.csv',rows);
  for(const result of results.transfer_results) {
    contrasts.push({model:result.model,source:result.source,target:result.target,protocol:result.signature.protocol,
      source_estimation_n:result.source_partition.estimation.length,target_estimation_n:result.target_partition.estimation.length,target_heldout_n:result.target_partition.heldout.length,
      ...Object.fromEntries(Object.entries(result.contrasts).flatMap(([name,metrics])=>Object.entries(metrics).map(([metric,value])=>[`${name}_${metric}_pp`,value]))),
      denominator:'Identical target Tr for all methods',uncertainty:'Point estimates; no CI',
      result_path:path.join(output,'transfer',result.model,result.source+'_to_'+result.target+'.json')});
    for(const [role,dataset,prior,partition] of [['source',result.source,result.source_prior,result.source_partition],['target',result.target,result.target_prior,result.target_partition]]) {
      // One unique dataset prior per model; it is reused in the reverse direction.
      if(priors.some(p=>p.model===result.model&&p.benchmark===dataset)) continue;
      for(let j=0;j<4;j++) priors.push({model:result.model,benchmark:dataset,label:'ABCD'[j],prior_probability:prior[j],estimation_n:partition.estimation.length,
        population_n:partition.n,alpha:partition.alpha,seed:partition.seed,protocol:result.signature.protocol,
        estimation_ids_sha256:sha(JSON.stringify(partition.estimation)),source_manifest:result.provenance[role].manifest,
        uncertainty:'One seed-42 point estimate; no interval'});
    }
  }
  await exportRows('tables/rq3_contrasts.csv',contrasts);
  await exportRows('tables/rq3_priors.csv',priors);
}
await exportRows('tables/input_representation_differences.csv',Object.entries(results.input_representation_audit??{}).flatMap(([benchmark,audit])=>audit.differences.map(r=>({benchmark,...r,reader:audit.reader}))));

if(updateLedger) {
  const ledgerPath=path.join(root,'experiment_evidence_ledger.csv');
  const original=await fs.readFile(ledgerPath,'utf8');
  const baseline=JSON.parse(await fs.readFile(path.join(input,'preservation_baseline.json'),'utf8'));
  if(sha(original)!==baseline.files['experiment_evidence_ledger.csv'].sha256) throw new Error('Root ledger changed since checksummed backup; preserve/review it first');
  const matrix=parseCSV(original);
  if(matrix.some(row=>row.length!==matrix[0].length)) throw new Error('Ragged ledger CSV');
  const columns=matrix[0];
  let oldRows=matrix.slice(1).filter(row=>row[0]).map(row=>Object.fromEntries(columns.map((key,i)=>[key,row[i]??''])));
  oldRows=oldRows.filter(row=>!String(row.experimental_block).startsWith('C:'));
  for(const row of oldRows) {
    if(row.experimental_block==='A: preserved six-model raw evaluation') {
      const verified=results.validation.find(v=>v.model===row.model&&v.dataset===row.benchmark);
      if(!verified) throw new Error('Missing validation row');
      row.verification_status=verified.status;
      row.hash_verification_status=verified.status==='verified_from_matching_raw_observations'?'Observation and manifest SHA256 verified; endpoints independently recomputed':
        'Summary hash/algebra and all five path-seeded partitions checked; '+(verified.manifest_hash_matches?'manifest hash verified':'manifest requires recovery')+'; full raw unavailable locally';
      if(verified.actual_observation_sha256) row.observation_hash=verified.actual_observation_sha256;
      if(verified.status==='verified_from_matching_raw_observations') {
        row.artifact_role='Preserved summary independently reproduced from matching full raw observations';
        row.observation_location=verified.observation_path;row.manifest_location=verified.manifest_path;
        row.actual_bytes=verified.actual_observation_bytes;
        row.permutation_coverage=`Complete ${row.benchmark==='csqa'?5:4} cyclic shifts; raw source shapes ${JSON.stringify(verified.diagnostics.permutation_matrix_shapes)}`;
        row.prior_estimation_evidence='All five original path-seeded partitions, priors and hybrid/heldout endpoints independently reproduced; '+path.join(output,'tables/calibration_draws.csv');
        row.available_metrics='Verified Default/CPA/hybrid PriDe accuracy, recall, RStd; label supports; five-draw SD/min/max; cyclic RStd; mean/SD/range of shift accuracy; semantic consistency; exact ties';
        row.missing_metrics=(row.benchmark==='medqa'||row.benchmark==='medmcqa')?'Persisted logit and truncation traces; resolved tokenizer commit; approved uncertainty intervals':'Resolved tokenizer commit; approved uncertainty intervals';
        row.population_sha256=verified.population_source.population_sha256??verified.population_source.normalized_sha256??'';
      }
      row.notes='Corrected RQ mapping retained. Original summary unchanged. '+(row.benchmark==='mmlu'?'Ten source None options executed as nan; see input_representation_audit.md. ':'')+'Current validation: '+path.join(output,'reproduction_validation.csv');
    }
  }
  for(const available of results.transfer_availability) {
    const row=Object.fromEntries(columns.map(key=>[key,'']));
    Object.assign(row,{experimental_block:'C: offline paired medical transfer',research_question:'RQ3',
      artifact_role:available.status==='completed_from_verified_raw'?'Verified paired transfer result':'Tested CPU implementation; empirical results blocked by raw recovery',
      model:available.model,benchmark:available.source+' -> '+available.target,protocol_version:'phase3b-medical-transfer-v1; current raw scores only',
      dataset_population:available.target_n,summary_location:available.status==='completed_from_verified_raw'?path.join(output,'transfer',available.model,available.source+'_to_'+available.target+'.json'):'',
      scoring_configuration:'Cached current raw 2K-slot scores; no model inference',permutation_coverage:'Four complete cycles required on both medical benchmarks',
      prior_estimation_evidence:'Separate source/target estimation; sorted namespaced IDs; seed 42; floor(.05*N); epsilon 1e-10; target Te='+available.target_estimation_n+'; common Tr='+available.target_heldout_n,
      available_metrics:available.status==='completed_from_verified_raw'?'Default, in-domain, transfer, CPA; paired accuracy/RStd contrasts':'None empirical; synthetic tests passed',
      missing_metrics:available.status==='completed_from_verified_raw'?'Approved confidence intervals':'All real paired outcomes; current medical raw pairs require recovery',
      verification_status:available.status,notes:'Phase 3B explicitly targets all six current models. Historical chat probabilities are excluded. '+path.join(output,'partition_policy.json')});
    if(available.status==='completed_from_verified_raw') {
      const result=results.transfer_results.find(r=>r.model===available.model&&r.source===available.source&&r.target===available.target);
      const summary=await fs.readFile(row.summary_location);
      row.summary_sha256=sha(summary);row.summary_bytes=summary.length;
      row.observation_location=result.provenance.target.observations;row.observation_hash=result.provenance.target.observations_sha256;row.expected_observation_hash=row.observation_hash;
      row.actual_bytes=(await fs.stat(row.observation_location)).size;row.manifest_location=result.provenance.target.manifest;
      row.hash_verification_status='Both source and target observations/manifests match saved SHA256; raw identities and numeric endpoints independently verified';
      row.population_sha256=result.provenance.target.source.population_sha256;
      row.notes+=' Source evidence: '+result.provenance.source.observations+' SHA256 '+result.provenance.source.observations_sha256+'. Both priors and paired predictions saved in result.';
    }
    oldRows.push(row);
  }
  // Save to a new temporary name, then replace only the authorized, backed-up ledger.
  const staged=ledgerPath+'.phase3b-new';
  await exportMatrix(staged,[columns,...oldRows.map(row=>columns.map(key=>cell(row[key])))]);
  await fs.rename(staged,ledgerPath);
  manifest.at(-1).path=ledgerPath;
}
await fs.writeFile(path.join(output,'table_export_validation.json'),JSON.stringify({engine:'node:fs RFC4180 CSV',files:manifest},null,2)+'\n');
await fs.writeFile(path.join(output,'tables','table_metadata.json'),JSON.stringify({protocol:'original-open-clm-raw-prompt-1536',modelPanel:results.validation.map(v=>v.model).filter((v,i,a)=>a.indexOf(v)===i),
  populations:Object.fromEntries(results.validation.map(v=>[v.dataset,v.population_n])),units:'Accuracy/recall %, RStd/differences pp; semantic consistency fraction',
  primaryDenominator:'Full retained D, hybrid PriDe CPA on De + correction on Dr; transfer uses identical Tr for all four methods',
  calibration:'Five path-seeded draws for preserved results, SD ddof=0; new transfer seed-42 independent source/target partitions',
  uncertainty:'No confidence intervals or new significance tests executed',sourceResults:path.join(input,'analysis_results.json'),
  verification:'CSV cells checked against the supplied derived JSON; this export does not revalidate raw observations.',files:manifest},null,2)+'\n');
console.log(JSON.stringify({csvOutputs:manifest.length,rows:manifest.map(r=>({path:r.path,rows:r.rows}))}));
