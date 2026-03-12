"""Pydantic models for CLI/YAML configuration validation."""
from __future__ import annotations
from typing import Optional, Tuple

from pydantic import BaseModel, Field, field_validator


class _HintVector(BaseModel):
    x: float
    y: float
    z: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


class SweepBase(BaseModel):
    isotope: str = Field(..., min_length=1)
    Ka: float = 0.0
    Kb: float = 0.0
    Kc: float = 0.0
    vQ_MHz: float = 0.0
    eta: float = 0.0
    phi: float = 0.0
    theta: float = 0.0
    psi: float = 0.0
    Hint_pas: Optional[_HintVector] = None
    B_axis: str = Field('z', pattern='^[xyzXYZ]$')

    @field_validator('vQ_MHz')
    def check_vq(cls, v):
        if v < 0:
            raise ValueError('vQ_MHz must be non-negative')
        return v


class FreqVsFieldConfig(SweepBase):
    B_min_T: float = 0.0
    B_max_T: float = 12.0
    B_points: int = Field(ge=2, default=400)

    @field_validator('B_max_T')
    def check_field_range(cls, v, info):
        min_val = info.data.get('B_min_T') if info.data else None
        if min_val is not None and v <= min_val:
            raise ValueError('B_max_T must be greater than B_min_T')
        return v


class ElevelsVsFieldConfig(FreqVsFieldConfig):
    pass


class FreqVsEtaConfig(BaseModel):
    isotope: str = Field(..., min_length=1)
    Ka: float = 0.0
    Kb: float = 0.0
    Kc: float = 0.0
    vQ_MHz: float = 0.0
    eta_min: float = 0.0
    eta_max: float = 1.0
    eta_points: int = Field(ge=2, default=200)
    phi: float = 0.0
    theta: float = 0.0
    psi: float = 0.0
    B0_T: float = 9.4
    B_axis: str = Field('z', pattern='^[xyzXYZ]$')
    Hint_pas: Optional[_HintVector] = None

    @field_validator('eta_max')
    def check_eta(cls, v, info):
        min_val = info.data.get('eta_min') if info.data else None
        if min_val is not None and v <= min_val:
            raise ValueError('eta_max must be greater than eta_min')
        return v


class FreqVsAngleConfig(BaseModel):
    isotope: str = Field(..., min_length=1)
    Ka: float = 0.0
    Kb: float = 0.0
    Kc: float = 0.0
    vQ_MHz: float = 0.0
    eta: float = 0.0
    H0_T: float = 9.4
    angle_name: str = Field('theta', pattern='^(phi|theta|psi)$')
    angle_min_deg: float = 0.0
    angle_max_deg: float = 90.0
    angle_points: int = Field(ge=2, default=361)
    fixed_phi_deg: float = 0.0
    fixed_theta_deg: float = 0.0
    fixed_psi_deg: float = 0.0
    B_axis: str = Field('z', pattern='^[xyzXYZ]$')
    Hint_pas: Optional[_HintVector] = None

    @field_validator('angle_max_deg')
    def check_angle_range(cls, v, info):
        min_val = info.data.get('angle_min_deg') if info.data else None
        if min_val is not None and v <= min_val:
            raise ValueError('angle_max_deg must be greater than angle_min_deg')
        return v


CMD_TO_MODEL = {
    'freq-vs-field': FreqVsFieldConfig,
    'elevels-vs-field': ElevelsVsFieldConfig,
    'freq-vs-eta': FreqVsEtaConfig,
    'freq-vs-angle': FreqVsAngleConfig,
}

__all__ = [
    'FreqVsFieldConfig',
    'ElevelsVsFieldConfig',
    'FreqVsEtaConfig',
    'FreqVsAngleConfig',
    'CMD_TO_MODEL',
]
