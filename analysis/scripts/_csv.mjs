// RFC 4180 CSV serialization using only the Node.js standard library.
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';

export const sha = value => crypto.createHash('sha256').update(value).digest('hex');
export const cell = (value, missing = '') => value === null || value === undefined
  ? missing : typeof value === 'object' ? JSON.stringify(value) : String(value);

export function parseCSV(text) {
  const rows = []; let row = [], value = '', quoted = false, closed = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') { value += '"'; i++; }
        else { quoted = false; closed = true; }
      } else value += ch;
    } else if (ch === ',' || ch === '\n' || ch === '\r') {
      row.push(value); value = ''; closed = false;
      if (ch !== ',') {
        if (ch === '\r' && text[i + 1] === '\n') i++;
        rows.push(row); row = [];
      }
    } else if (ch === '"' && value === '' && !closed) quoted = true;
    else {
      if (closed || ch === '"') throw new Error('Malformed CSV quoting');
      value += ch;
    }
  }
  if (quoted) throw new Error('Unclosed CSV field');
  if (row.length || value || closed) { row.push(value); rows.push(row); }
  return rows;
}

export function serializeCSV(matrix, missing = '') {
  if (!matrix.length || !matrix[0].length) throw new Error('Empty CSV matrix');
  const values = matrix.map(row => row.map(value => cell(value, missing)));
  if (values.some(row => row.length !== values[0].length)) throw new Error('Ragged CSV matrix');
  const csv = values.map(row => row.map(value => '"' + value.replaceAll('"', '""') + '"').join(',')).join('\r\n') + '\r\n';
  if (JSON.stringify(parseCSV(csv)) !== JSON.stringify(values)) throw new Error('CSV round-trip differs');
  return csv;
}

export async function writeMatrix(filename, matrix, missing = '') {
  const csv = serializeCSV(matrix, missing);
  await fs.mkdir(path.dirname(filename), {recursive: true});
  try { await fs.writeFile(filename, csv, {flag: 'wx'}); }
  catch (error) {
    if (error.code !== 'EEXIST') throw error;
    if (await fs.readFile(filename, 'utf8') !== csv) throw new Error(`Refusing to overwrite different CSV: ${filename}`);
  }
  return {path: filename, rows: matrix.length - 1, columns: matrix[0].length,
    sha256: sha(csv), all_cells_verified: true};
}
