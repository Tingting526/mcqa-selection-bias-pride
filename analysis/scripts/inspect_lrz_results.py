"""Read Slurm status and saved results; never submit or cancel jobs."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT = Path.cwd() if __file__ == '<stdin>' else Path(__file__).resolve().parents[1]


def command(args):
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=45)
        return dict(command=args, returncode=result.returncode,
                    stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(command=args, error=str(error), stdout='')


def inspect(directory):
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  registered_jobs=[], pipelines=[], reference_runs=[], errors=[])
    report['sacct'] = command(['sacct', '--starttime', '2026-09-29', '--allocations',
                              '--parsable2', '--format=JobID,JobName,State,ExitCode,Elapsed'])
    report['squeue'] = command(['squeue', '--me', '-o', '%.12i %.24j %.12T %.12M %.70R'])
    campaigns = ROOT/'outputs/campaigns'
    for pattern in ('**/campaign.json', '**/gemma_jobs.json', '**/llama_jobs.json', '**/serial_jobs.json'):
        for path in sorted(campaigns.glob(pattern)):
            try:
                registry = json.loads(path.read_text())
                for entry in registry.get('entries', registry.get('groups', [])):
                    if entry.get('job_id'):
                        report['registered_jobs'].append(dict(source=str(path), **entry))
            except (ValueError, OSError) as error:
                report['errors'].append(f'{path}: {error}')
    roots = [campaigns, ROOT/'outputs/lrz-results/full-20261009',
             ROOT/'outputs/lrz-results/pilots-20261008']
    for root in roots:
        for path in sorted(root.glob('**/pipeline.json')):
            try:
                state = json.loads(path.read_text())
                if state.get('config', {}).get('scorer') != 'real':
                    continue
                comparison = path.with_name('comparison.csv')
                state.update(source=str(path), comparison=list(csv.DictReader(
                    comparison.open())) if comparison.exists() else [])
                report['pipelines'].append(state)
            except (ValueError, OSError) as error:
                report['errors'].append(f'{path}: {error}')
    for path in sorted((ROOT/'outputs/zheng_selection_bias').glob('*/manifest.json')):
        try:
            state = json.loads(path.read_text())
            summary = path.with_name('summary.csv')
            state.update(source=str(path), zero_shot_summary=[r for r in
                csv.DictReader(summary.open()) if r['shots'] == '0'] if summary.exists() else [])
            report['reference_runs'].append(state)
        except (ValueError, OSError, KeyError) as error:
            report['errors'].append(f'{path}: {error}')
    report['failed_job_logs'] = []
    report['serial_recovery'] = []
    for path in sorted(campaigns.glob('**/worker_status.json')):
        try:
            report['serial_recovery'].append(dict(source=str(path), **json.loads(path.read_text())))
        except (ValueError, OSError) as error:
            report['errors'].append(f'{path}: {error}')
    report['canonical_pride'] = []
    report['canonical_pride_state'] = {}
    reproduction = ROOT/'outputs/reproductions/zheng-pride-0shot-v1'
    for path in sorted((reproduction/'results').glob('*/*/result.json')):
        try:
            report['canonical_pride'].append(dict(source=str(path), **json.loads(path.read_text())))
        except (ValueError, OSError) as error:
            report['errors'].append(f'{path}: {error}')
    for filename in ('inventory.json', 'jobs.json'):
        path = reproduction/filename
        if path.exists():
            report['canonical_pride_state'][filename] = json.loads(path.read_text())
    for path in sorted((reproduction/'workers').glob('*/worker_status.json')):
        report['serial_recovery'].append(dict(source=str(path), **json.loads(path.read_text())))
    for row in csv.DictReader(report['sacct']['stdout'].splitlines(), delimiter='|'):
        if not row.get('JobName', '').startswith('mcq-'):
            continue
        if row.get('State', '').split()[0] in {'COMPLETED', 'PENDING', 'RUNNING'}:
            continue
        job = row.get('JobID', '')
        if not job.isdigit():
            continue
        logs = {}
        for suffix in ('err', 'out'):
            for path in (ROOT/'outputs/logs/slurm').glob(f'*-{job}.{suffix}'):
                logs[path.name] = '\n'.join(path.read_text(errors='replace').splitlines()[-45:])
        report['failed_job_logs'].append(dict(job=row, logs=logs))
    if directory is None:
        print(json.dumps(report, indent=2))
        return report
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'report.json').write_text(json.dumps(report, indent=2))
    lines = ['=== SLURM ACCOUNTING ===', report['sacct']['stdout'],
             '=== CURRENT QUEUE ===', report['squeue']['stdout'], '=== SAVED RESULTS ===']
    seen = set()
    for state in report['pipelines']:
        config = state['config']
        pipeline_id = Path(state['source']).parent.name
        if pipeline_id in seen:
            continue
        seen.add(pipeline_id)
        scope = 'FULL' if config.get('limit') is None else f'PILOT limit={config["limit"]}'
        lines.append(f'{config["model"]} / {config["dataset"]}: {scope}, '
                     f'{state.get("status", "incomplete")}, comparison rows={len(state["comparison"])}')
    for state in report['reference_runs']:
        config = state['config']
        scope = 'FULL' if config.get('limit') is None else f'PILOT limit={config["limit"]}'
        lines.append(f'{config["model"]} / {config["dataset"]}: {scope}, '
                     f'{state.get("status", "incomplete")}, 0-shot rows={len(state["zero_shot_summary"])}')
    lines += ['=== FAILED JOB LOGS ===', json.dumps(report['failed_job_logs'], indent=2),
              '=== COLLECTION ERRORS ===', '\n'.join(report['errors']) or '(none)']
    (directory/'report.txt').write_text('\n'.join(lines))
    print('\n'.join(lines[:5]))
    print(f'Reports saved in {directory}')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path,
                        default=ROOT/'outputs/lrz-results/audit-20261015')
    parser.add_argument('--stdout-json', action='store_true',
                        help='Print report without writing files on LRZ; supports python - over SSH.')
    args = parser.parse_args()
    inspect(None if args.stdout_json else args.directory)
