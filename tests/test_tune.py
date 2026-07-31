import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("tune_ccas", ROOT / "scripts" / "tune_ccas.py")
tune = importlib.util.module_from_spec(_spec)
sys.modules["tune_ccas"] = tune
_spec.loader.exec_module(tune)


def test_phase_a_has_six_unique_candidates_per_dataset():
    for dataset in tune.DATASETS:
        rows = tune.candidates(dataset)
        assert len(rows) == 6
        assert len({row[2] for row in rows}) == 6
        assert all("hpoA_lr" in row[2] for row in rows)


def test_warmup_is_about_ten_percent_of_each_training_schedule():
    assert tune.WARMUP == {"rxrx1": 3, "camelyon17": 1}
