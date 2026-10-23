"""The released 30-min series pass the physical bounds whenever a calibration or test block is read."""
from pathlib import Path

import pytest

from amazon_graybox import experiments

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


@pytest.mark.skipif(not (PROCESSED / "jaru_2008.csv").exists(), reason="data/processed not present")
@pytest.mark.parametrize("season", ["dry", "wet"])
def test_processed_blocks_pass_physical_bounds(season):
    train, val, test = experiments.calibration_split(PROCESSED, season)  # raises PhysicalBoundError on a violation
    assert len(train) > len(val) > 0 and len(test) > 0
    sets = experiments.test_sets(PROCESSED, season)
    assert {"jaru2009", "jaru2010"} <= set(sets)
