import pytest
from pydantic import ValidationError

from pysimnmr.config_schema import FreqVsFieldConfig, FreqVsEtaConfig


def test_freq_vs_field_range_validation():
    with pytest.raises(ValidationError):
        FreqVsFieldConfig(isotope='1H', B_min_T=5.0, B_max_T=4.0)


def test_freq_vs_eta_points_validation():
    cfg = FreqVsEtaConfig(isotope='1H', eta_min=0.0, eta_max=0.2)
    assert cfg.eta_points >= 2
    with pytest.raises(ValidationError):
        FreqVsEtaConfig(isotope='1H', eta_min=0.5, eta_max=0.1)
