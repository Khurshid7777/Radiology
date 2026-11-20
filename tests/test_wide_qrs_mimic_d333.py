"""D333: a measured QRS >= 120 ms makes discordant ST elevation judged as under a block (Smith-modified Sgarbossa)."""
from cardiolens.layer4_findings import rules


def test_threshold_is_the_wide_qrs_bound():
    assert rules.WIDE_QRS_MIMIC_MS == 120.0
