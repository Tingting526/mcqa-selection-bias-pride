#!/usr/bin/env python3
"""Read-only LRZ pull into a NEW isolated directory, followed by SHA256 checks.

Uses only the Python standard library and rsync/ssh. No model libraries or remote
writes. A failed/partial copy still produces a machine-readable verification report.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
REMOTE_ROOT="/dss/dsshome1/<group>/<lrz-user>/analysis/"
HOST="<lrz-user>@login.ai.lrz.de"


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def load_expected(manifest,list_path):
    expected=json.loads(manifest.read_text())
    relative={}
    for absolute,sha in expected.items():
        if not absolute.startswith(REMOTE_ROOT): raise ValueError("Path outside documented project")
        path=Path(absolute[len(REMOTE_ROOT):])
        if path.is_absolute() or ".." in path.parts or not path.parts: raise ValueError("Unsafe relative path")
        if len(sha)!=64 or any(c not in "0123456789abcdef" for c in sha): raise ValueError("Invalid expected SHA256")
        relative[path.as_posix()]=sha
    listed=list_path.read_text().splitlines()
    if len(set(listed))!=len(listed) or set(listed)!=set(relative):
        raise ValueError("Recovery list and expected hash manifest disagree")
    return relative


def verify(destination,expected):
    results=[]
    for relative,sha in sorted(expected.items()):
        path=destination/relative
        item=dict(relative_path=relative,local_path=str(path),expected_sha256=sha)
        if not path.is_file():
            item.update(status="MISSING",actual_sha256=None,bytes=None)
        elif path.is_symlink():
            item.update(status="REJECTED_SYMLINK",actual_sha256=None,bytes=None)
        else:
            actual=digest(path)
            item.update(status="VERIFIED" if actual==sha else "HASH_MISMATCH",actual_sha256=actual,bytes=path.stat().st_size)
        results.append(item)
    return results


def recover(destination,expected,list_path,*,verify_only=False,run_command=subprocess.run):
    transfer=dict(attempted=False,returncode=None)
    if not verify_only:
        if destination.exists() and any(destination.iterdir()):
            raise ValueError("Refusing to copy into nonempty directory; select a NEW isolated destination")
        destination.mkdir(parents=True,exist_ok=True)
        command=["rsync","-av","--partial-dir=.rsync-partial","--delay-updates","--timeout=60",
                 "-e","ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=yes",
                 "--files-from="+str(list_path),HOST+":"+REMOTE_ROOT,str(destination)+"/"]
        transfer["attempted"]=True
        try:
            result=run_command(command,check=False,capture_output=True,text=True)
            # Do not echo remote output or potentially credential-bearing logs.
            transfer.update(returncode=result.returncode,
                            diagnostic="copy completed" if result.returncode==0 else "rsync failed or transfer incomplete; verify statuses below; no remote absence inferred")
        except OSError as error:
            transfer.update(returncode=127,diagnostic=type(error).__name__+": rsync/ssh could not be executed")
    files=verify(destination,expected)
    complete=all(r["status"]=="VERIFIED" for r in files)
    return dict(created_at=dt.datetime.now(dt.timezone.utc).isoformat(),destination=str(destination),
                remote_host=HOST,remote_root=REMOTE_ROOT,remote_writes=False,transfer=transfer,
                complete=complete,expected_files=len(files),verified_files=sum(r["status"]=="VERIFIED" for r in files),files=files)


def main():
    stamp=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base=ROOT/"analysis/outputs/phase3b_20261017"
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination",type=Path,default=base/"recovered_lrz"/stamp)
    parser.add_argument("--expected-manifest",type=Path,default=ROOT/"tmp/phase3a_20261017/expected_remote_hashes.json")
    parser.add_argument("--files-list",type=Path,default=ROOT/"tmp/phase3a_20261017/recovery_files.txt")
    parser.add_argument("--report",type=Path)
    parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args()
    expected=load_expected(args.expected_manifest,args.files_list)
    destination=args.destination.absolute()
    report=args.report or destination.parent/(destination.name+"_verification_"+stamp+".json")
    if report.exists(): raise FileExistsError("Select a new verification report path")
    result=recover(destination,expected,args.files_list,verify_only=args.verify_only)
    result.update(expected_manifest=str(args.expected_manifest),expected_manifest_sha256=digest(args.expected_manifest),
                  files_list=str(args.files_list),files_list_sha256=digest(args.files_list))
    report.parent.mkdir(parents=True,exist_ok=True)
    with report.open("x") as stream: json.dump(result,stream,indent=2); stream.write("\n")
    print(json.dumps(dict(report=str(report),destination=str(destination),complete=result["complete"],verified_files=result["verified_files"],expected_files=result["expected_files"])))
    return 0 if result["complete"] and result["transfer"]["returncode"] in (None,0) else 2


if __name__=="__main__": sys.exit(main())
