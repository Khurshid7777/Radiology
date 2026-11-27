# P-wave detector (D295) — NOT wired into the pipeline

`p_cnn_d295_it2.onnx` (150 KB): 1-D CNN, one lead at 500 Hz band 0.5-40 Hz scaled by its robust SD -> per-sample logit
"inside a P wave". Trained by `tools/p_detector/cnn.py` (iteration 2, QTDB crop share 0.4, 5 000 steps, seed 295) on
even LUDB ids (except ids % 4 == 0) and even QTDB records (except every 4th) — open data (LUDB, QT Database: ODC-By).
P peaks = centres of regions with probability >= 0.6 (threshold chosen on the validation part).
Test (odd ids, seen twice — DECISIONS D295): LUDB II recall 0.964 / precision 0.993 (p_chain 0.86 / 0.99); QTDB recall 0.79;
false "P" on LUDB AF / flutter 3.9 per record (p_chain 2.5). Not accepted as a general detector; see D295.
