import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

path=Path(__file__).resolve().parents[1]/"scripts/recover_phase3b_evidence.py"
spec=importlib.util.spec_from_file_location("phase3b_recovery",path)
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_missing_partial_corrupt_and_complete_files(tmp_path):
    expected={"nested/records.jsonl":hashlib.sha256(b"complete").hexdigest()}
    assert module.verify(tmp_path,expected)[0]["status"]=="MISSING"
    file=tmp_path/"nested/records.jsonl"; file.parent.mkdir(); file.write_bytes(b"comp")
    assert module.verify(tmp_path,expected)[0]["status"]=="HASH_MISMATCH"
    file.write_bytes(b"complete")
    assert module.verify(tmp_path,expected)[0]["status"]=="VERIFIED"


def test_copy_is_pull_only_and_failure_still_reports_missing(tmp_path):
    calls=[]
    def fake_run(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=255)
    result=module.recover(tmp_path/"new",{"a":"0"*64},tmp_path/"list",run_command=fake_run)
    assert not result["complete"] and result["files"][0]["status"]=="MISSING"
    command=calls[0]
    assert command[-2]==module.HOST+":"+module.REMOTE_ROOT
    assert command[-1]==str(tmp_path/"new")+"/"
    assert not any("--delete" in arg for arg in command)
    assert not result["remote_writes"]


def test_refuse_overwrite_or_symlink_and_verify_only_no_network(tmp_path):
    dest=tmp_path/"existing"; dest.mkdir(); (dest/"original").write_text("keep")
    with pytest.raises(ValueError,match="nonempty"):
        module.recover(dest,{"a":"0"*64},tmp_path/"list")
    link=dest/"a"; link.symlink_to(dest/"original")
    result=module.recover(dest,{"a":"0"*64},tmp_path/"list",verify_only=True,
                          run_command=lambda *a,**k:pytest.fail("Network called"))
    assert result["files"][0]["status"]=="REJECTED_SYMLINK"
    assert (dest/"original").read_text()=="keep"


def test_manifest_paths_and_checksums_cross_checked(tmp_path):
    import json
    manifest=tmp_path/"hashes.json"; listing=tmp_path/"list.txt"
    manifest.write_text(json.dumps({module.REMOTE_ROOT+"a":"0"*64}))
    listing.write_text("a\n")
    assert module.load_expected(manifest,listing)=={"a":"0"*64}
    listing.write_text("other\n")
    with pytest.raises(ValueError,match="disagree"): module.load_expected(manifest,listing)
    manifest.write_text(json.dumps({module.REMOTE_ROOT+"../bad":"0"*64}))
    with pytest.raises(ValueError,match="Unsafe"): module.load_expected(manifest,listing)
