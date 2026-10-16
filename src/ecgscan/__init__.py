"""ecgscan — paper-ECG photo/scan digitisation pipeline (docs/scanner).

A web-free library: image + metadata in, ``ScanResult`` out. The CardioLens
signal contract is reached only through ``ecgscan.contract.cardiolens_adapter``
(imported lazily), so the core does not depend on the interpretation layers
(DECISIONS SD-002).
"""

__version__ = "0.1.0"
PIPELINE_VERSION = __version__
SCHEMA_VERSION = "1.0.0"
