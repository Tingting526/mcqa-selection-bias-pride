import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { cell, parseCSV, serializeCSV, writeMatrix } from '../scripts/_csv.mjs';

test('CSV preserves precision, quotes, newlines, Unicode and missing values', () => {
  const matrix = [['name', 'value'], ['A, B', 'a"b\nc'], ['μ', 0.12345678901234568],
    ['missing', null], ['array', [0.1, 0.2]], ['boolean', false], ['empty', '']];
  assert.deepEqual(parseCSV(serializeCSV(matrix)), matrix.map(row => row.map(value => cell(value))));
  assert.deepEqual(parseCSV(serializeCSV([['missing', null]], 'NA')), [['missing', 'NA']]);
  assert.throws(() => parseCSV('"unclosed'), /Unclosed/);
  assert.throws(() => serializeCSV([['a', 'b'], ['c']]), /Ragged/);
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
