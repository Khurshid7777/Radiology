"""Copy the RU/UZ strings (single source: configs/scanner/i18n, SD-016) into the app's assets."""
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[3]
for lang in ("ru", "uz"):
    shutil.copyfile(root / "configs/scanner/i18n" / f"{lang}.json",
                    Path(__file__).resolve().parents[1] / "assets/i18n" / f"{lang}.json")
