"""D263: a scanner read flagged PR_UNSTABLE gets no PR findings and a 'PR approximate' note."""
from cardiolens.contracts.findings import Finding, FindingSet
from cardiolens.layer1_ingestion.synthetic import generate_synthetic_ecg
from cardiolens.pipeline import _pr_unstable


def _sig(notes):
    sig = generate_synthetic_ecg()
    prov = sig.provenance.model_copy(update={"digitized": True, "notes": notes})
    return sig.model_copy(update={"provenance": prov})


def _fs():
    return FindingSet(findings=[Finding(code=c, label_ru=c, label_uz=c, confidence=0.8)
                                for c in ("first_degree_av_block", "pr_borderline", "rbbb")])


def test_flagged_read_withholds_pr_findings():
    fs, audit = _fs(), []
    _pr_unstable(_sig(["ecgscan scan_id=x status=needs_confirmation pipeline=1", "warnings: PR_UNSTABLE"]), fs, audit)
    assert fs.codes() == {"rbbb", "pr_approximate"} and "D263" in audit[0]


def test_unflagged_reads_unchanged():
    for notes in (["ecgscan scan_id=x status=ok pipeline=1", "warnings: QRS_UNSTABLE"], []):
        fs = _fs()
        _pr_unstable(_sig(notes), fs, [])
        assert fs.codes() == {"first_degree_av_block", "pr_borderline", "rbbb"}
