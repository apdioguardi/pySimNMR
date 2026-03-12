import pytest

from pysimnmr.powder_field_simulation import PowderFieldConfig
from pysimnmr.config_aliases import normalize_powder_field_config


BASE_CFG = {
    'isotope_list': ['63Cu'],
    'site_multiplicity_list': [1.0],
    'Ka_list': [0.0],
    'Kb_list': [0.0],
    'Kc_list': [0.0],
    'vc_list': [3.0],
    'eta_list': [0.0],
    'va_list': [None],
    'vb_list': [None],
    'f0': 10.0,
    'min_field': 0.0,
    'max_field': 1.0,
    'n_field_points': 5,
    'convolution_function_list': ['gauss'],
    'conv_FWHM_list': [0.01],
    'conv_vQ_FWHM_list': [0.0],
}


def _make_cfg(**overrides) -> PowderFieldConfig:
    cfg = dict(BASE_CFG)
    cfg.update(overrides)
    return PowderFieldConfig(**cfg)


def test_powder_field_config_derives_eta_from_tensor_components():
    va, vb = -2.0, -1.0
    cfg = _make_cfg(va_list=[va], vb_list=[vb], eta_list=None)
    assert pytest.approx(cfg.eta_list[0]) == pytest.approx((va - vb) / cfg.vc_list[0])


def test_powder_field_config_derives_vc_from_tensor_components():
    va, vb = -2.5, -1.5
    cfg = _make_cfg(va_list=[va], vb_list=[vb], vc_list=None, eta_list=None)
    assert pytest.approx(cfg.vc_list[0]) == pytest.approx(-(va + vb))
    assert pytest.approx(cfg.eta_list[0]) == pytest.approx((va - vb) / (-(va + vb)))


def test_powder_field_config_requires_eta_or_tensor_with_vq():
    with pytest.raises(ValueError):
        _make_cfg(
            eta_list=None,
            va_list=[None],
            vb_list=[None],
            vc_list=None,
            vQ_list=[2.0],
        )


def test_normalize_powder_field_config_hint_vector_expansion():
    cfg = {
        'isotope_list': ['1H', '13C'],
        'site_multiplicity_list': [1.0, 1.0],
        'Ka_list': [0.0, 0.0],
        'Kb_list': [0.0, 0.0],
        'Kc_list': [0.0, 0.0],
        'vc_list': [3.0, 3.0],
        'eta_list': [0.0, 0.5],
        'Hint_list': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        'f0': 10.0,
        'min_field': 0.0,
        'max_field': 1.0,
        'n_field_points': 10,
        'convolution_function_list': ['gauss'],
        'conv_FWHM_list': [0.01],
        'conv_vQ_FWHM_list': [0.0],
    }
    norm = normalize_powder_field_config(cfg)
    assert norm['Hinta_list'] == [1.0, 4.0]
    assert norm['Hintb_list'] == [2.0, 5.0]
    assert norm['Hintc_list'] == [3.0, 6.0]
