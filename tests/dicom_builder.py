"""Synthetic DICOM waveform ECG files for the tests (D94, D96)."""

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

ECG_12 = "1.2.840.10008.5.1.4.1.1.9.1.1"
LEAD_CODES = {"I": "1", "II": "2", "III": "61", "aVR": "62", "aVL": "63", "aVF": "64",
              "V1": "3", "V2": "4", "V3": "5", "V4": "6", "V5": "7", "V6": "8"}


def code(value: str, scheme: str, meaning: str) -> Dataset:
    c = Dataset()
    c.CodeValue, c.CodingSchemeDesignator, c.CodeMeaning = value, scheme, meaning
    return c


def group(sig_uv: np.ndarray, names: list[str], fs: float, *, unit: str = "uV", sens: float = 2.5,
           baseline: float = 0.0, correction: float = 1.0, label: str = "RHYTHM", by_code: bool = False) -> Dataset:
    g = Dataset()
    g.MultiplexGroupLabel = label
    g.WaveformOriginality = "ORIGINAL"
    g.NumberOfWaveformChannels = len(names)
    g.NumberOfWaveformSamples = sig_uv.shape[1]
    g.SamplingFrequency = fs
    g.WaveformBitsAllocated = 16
    g.WaveformSampleInterpretation = "SS"
    scale = {"uV": 1.0, "mV": 1000.0}[unit]
    chans = []
    for n in names:
        ch = Dataset()
        meaning = "" if by_code else f"Lead {n}"
        ch.ChannelSourceSequence = Sequence([code(f"5.6.3-9-{LEAD_CODES[n]}", "SCPECG", meaning)])
        ch.ChannelSensitivity = sens
        ch.ChannelSensitivityUnitsSequence = Sequence([code(unit, "UCUM", unit)])
        ch.ChannelSensitivityCorrectionFactor = correction
        ch.ChannelBaseline = baseline
        ch.WaveformBitsStored = 16
        chans.append(ch)
    g.ChannelDefinitionSequence = Sequence(chans)
    counts = np.round(sig_uv / scale / (sens * correction) - baseline).astype("<i2")
    g.WaveformData = counts.T.tobytes()                      # interleaved: sample-major
    return g


def write(path, groups: list[Dataset], *, sop: str = ECG_12) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID = sop, generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID, ds.Modality = sop, meta.MediaStorageSOPInstanceUID, "ECG"
    ds.PatientName, ds.PatientID, ds.PatientBirthDate = "Ivanov^Ivan", "MRN-12345", "19800101"
    ds.PatientAge, ds.PatientSex = "045Y", "F"
    ds.WaveformSequence = Sequence(groups)
    ds.save_as(str(path), enforce_file_format=True)


def dataset(groups: list[Dataset], *, sop: str = ECG_12) -> Dataset:
    """The same object as ``write`` saves, in memory (for a network send)."""
    ds = Dataset()
    ds.SOPClassUID, ds.SOPInstanceUID, ds.Modality = sop, generate_uid(), "ECG"
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.PatientName, ds.PatientID, ds.PatientBirthDate = "Ivanov^Ivan", "MRN-12345", "19800101"
    ds.PatientAge, ds.PatientSex = "045Y", "F"
    ds.WaveformSequence = Sequence(groups)
    return ds
