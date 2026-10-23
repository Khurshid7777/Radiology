"""D148: the limb QRS amplitude (low voltage) is measured on the 0.5-150 Hz copy for recorded
signals; a digitised trace stays on the measuring band (its pixel steps are not amplitude)."""

import numpy as np

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer3_measure.measure import measure
from cardiolens.pipeline import run_pipeline


def test_the_wide_band_copy_measures_a_taller_qrs_than_the_40_hz_pass():
    sig = generate_synthetic_ecg(seed=2)
    res = run_pipeline(sig, mode="A")
    from cardiolens.config import load_app_config
    from cardiolens.layer2_preprocess.filters import preprocess

    pp = load_app_config().preprocessing
    narrow = preprocess(sig.model_copy(deep=True), mains_frequency_hz=50, bandpass_low_hz=pp.bandpass_low_hz,
                        bandpass_high_hz=pp.bandpass_high_hz, target_sampling_rate_hz=pp.target_sampling_rate_hz)
    m40 = measure(narrow)
    assert res.measurements.qrs_amplitude_limb_uv > m40.qrs_amplitude_limb_uv


def test_a_misaligned_amplitude_copy_is_ignored():
    sig = generate_synthetic_ecg(seed=3)
    short = sig.model_copy(update={"signal": np.asarray(sig.signal)[:, :1000]})
    assert measure(sig, amplitude_sig=short).qrs_amplitude_limb_uv == measure(sig).qrs_amplitude_limb_uv
