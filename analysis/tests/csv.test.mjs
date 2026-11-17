import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { cell, parseCSV, serializeCSV, writeMatrix, sha } from '../scripts/_csv.mjs';

test('CSV preserves precision, quotes, newlines, Unicode and missing values', () => {
  const matrix = [['name', 'value'], ['A, B', 'a"b\nc'], ['μ', 0.12345678901234568],
    ['missing', null], ['array', [0.1, 0.2]], ['boolean', false], ['empty', '']];
  assert.deepEqual(parseCSV(serializeCSV(matrix)), matrix.map(row => row.map(value => cell(value))));
  assert.deepEqual(parseCSV(serializeCSV([['missing', null]], 'NA')), [['missing', 'NA']]);
  assert.throws(() => parseCSV('"unclosed'), /Unclosed/);
  assert.throws(() => serializeCSV([['a', 'b'], ['c']]), /Ragged/);
});

test('primary CSV export includes its referenced transfer evidence and protects existing results', async () => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'mcqa-transfer-export-test-'));
  const input = fileURLToPath(new URL('../outputs/phase3b_20261018/verified/', import.meta.url));
  const script = fileURLToPath(new URL('../scripts/export_phase3b_tables.mjs', import.meta.url));
  const sourcePath = path.join(input, 'analysis_results.json');
  const original = await fs.readFile(sourcePath, 'utf8');
  const results = JSON.parse(original).transfer_results;
  const run = () => promisify(execFile)(process.execPath, [script, input, directory]);
  try {
    const report = JSON.parse((await run()).stdout);
    assert.equal(report.csvOutputs, 14);
    assert.equal(report.transferOutputs, 12);
    const [columns, ...rows] = parseCSV(await fs.readFile(path.join(directory, 'tables/rq3_contrasts.csv'), 'utf8'));
    const manifest = JSON.parse(await fs.readFile(path.join(directory, 'table_export_validation.json'), 'utf8'));
    assert.equal(rows.length, results.length);
    assert.equal(manifest.transferFiles.length, results.length);
    for (const values of rows) {
      const row = Object.fromEntries(columns.map((key, i) => [key, values[i]]));
      const expected = results.find(result => result.model === row.model && result.source === row.source && result.target === row.target);
      assert.ok(expected);
      const content = await fs.readFile(row.result_path, 'utf8');
      assert.deepEqual(JSON.parse(content), expected);
      assert.equal(manifest.transferFiles.find(file => file.path === row.result_path).sha256, sha(content));
    }
    await run();
    const firstPath = manifest.transferFiles[0].path;
    const first = JSON.parse(await fs.readFile(firstPath, 'utf8'));
    const reformatted = JSON.stringify(first, null, 4);
    await fs.writeFile(firstPath, reformatted);
    await run();
    assert.equal(await fs.readFile(firstPath, 'utf8'), reformatted);
    const changed = JSON.stringify({...first, source: 'changed'});
    await fs.writeFile(firstPath, changed);
    await assert.rejects(run(), /Refusing to overwrite different transfer result/);
    assert.equal(await fs.readFile(firstPath, 'utf8'), changed);
    assert.equal(await fs.readFile(sourcePath, 'utf8'), original);
  } finally {
    await fs.rm(directory, {recursive: true, force: true});
  }
});

test('re-export is idempotent and refuses to overwrite changed evidence', async () => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'mcqa-csv-test-'));
  try {
    const file = path.join(directory, 'table.csv');
    const matrix = [['metric'], [0.12345678901234568]];
    const first = await writeMatrix(file, matrix);
    assert.equal((await writeMatrix(file, matrix)).sha256, first.sha256);
    await assert.rejects(writeMatrix(file, [['metric'], [99]]), /Refusing to overwrite/);
    assert.deepEqual(parseCSV(await fs.readFile(file, 'utf8')), [['metric'], ['0.12345678901234568']]);
  } finally {
    await fs.rm(directory, {recursive: true, force: true});
  }
});
