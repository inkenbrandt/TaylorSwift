"""
Tests for eccospectra.constants — physical constants, surface types,
quality thresholds, and roughness helper functions.
"""

import pytest

from eccospectra.constants import (
    K_VON_KARMAN,
    G0,
    R_GAS,
    CP_DRY_AIR,
    L_VAPORIZATION,
    MOLAR_MASS,
    R_SPECIFIC,
    T_ZERO_C,
    P_REFERENCE,
    SurfaceType,
    Hemisphere,
    ROUGHNESS_LENGTH,
    DISPLACEMENT_RATIO,
    QualityThreshold,
    ProcessingConfig,
    ErrorCode,
    UNIT_CONVERSION,
    get_displacement_height,
    get_roughness_length,
)


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

class TestPhysicalConstants:
    def test_von_karman_value(self):
        assert K_VON_KARMAN == pytest.approx(0.40)

    def test_gravity_value(self):
        assert G0 == pytest.approx(9.80665)

    def test_gas_constant_value(self):
        assert R_GAS == pytest.approx(8.3144598)

    def test_cp_dry_air_positive(self):
        assert CP_DRY_AIR > 0

    def test_latent_heat_vaporization_positive(self):
        assert L_VAPORIZATION > 0

    def test_t_zero_celsius(self):
        assert T_ZERO_C == pytest.approx(273.15)

    def test_p_reference_value(self):
        assert P_REFERENCE == pytest.approx(101.325)

    def test_molar_mass_keys(self):
        assert "co2" in MOLAR_MASS
        assert "h2o" in MOLAR_MASS
        assert "air_dry" in MOLAR_MASS

    def test_molar_mass_co2_approx(self):
        assert MOLAR_MASS["co2"] == pytest.approx(0.044010, rel=1e-3)

    def test_r_specific_keys(self):
        assert "dry_air" in R_SPECIFIC
        assert "water_vapor" in R_SPECIFIC


# ---------------------------------------------------------------------------
# SurfaceType enum
# ---------------------------------------------------------------------------

class TestSurfaceType:
    def test_all_members_present(self):
        members = {s.name for s in SurfaceType}
        assert "CROP" in members
        assert "FOREST" in members
        assert "WATER" in members
        assert "URBAN" in members

    def test_integer_values(self):
        assert int(SurfaceType.CROP) == 1
        assert int(SurfaceType.GRASS) == 2

    def test_roughness_length_has_all_types(self):
        for st in SurfaceType:
            assert st in ROUGHNESS_LENGTH

    def test_displacement_ratio_has_all_types(self):
        for st in SurfaceType:
            assert st in DISPLACEMENT_RATIO

    def test_displacement_ratios_in_unit_interval(self):
        for ratio in DISPLACEMENT_RATIO.values():
            assert 0.0 <= ratio <= 1.0


# ---------------------------------------------------------------------------
# Hemisphere enum
# ---------------------------------------------------------------------------

class TestHemisphere:
    def test_north_positive(self):
        assert int(Hemisphere.NORTH) == 1

    def test_south_negative(self):
        assert int(Hemisphere.SOUTH) == -1


# ---------------------------------------------------------------------------
# QualityThreshold
# ---------------------------------------------------------------------------

class TestQualityThreshold:
    def test_rn_threshold_keys(self):
        assert "high_quality" in QualityThreshold.RN_THRESHOLD
        assert "moderate_quality" in QualityThreshold.RN_THRESHOLD

    def test_valid_range_wind_speed(self):
        lo, hi = QualityThreshold.VALID_RANGE["wind_speed"]
        assert lo < 0 < hi

    def test_valid_range_co2(self):
        lo, hi = QualityThreshold.VALID_RANGE["co2"]
        assert lo > 0 and hi > lo

    def test_signal_strength_bounds(self):
        for val in QualityThreshold.SIGNAL_STRENGTH.values():
            assert 0.0 < val < 1.0


# ---------------------------------------------------------------------------
# ProcessingConfig
# ---------------------------------------------------------------------------

class TestProcessingConfig:
    def test_averaging_interval_positive(self):
        assert ProcessingConfig.AVERAGING_INTERVAL > 0

    def test_despike_z_threshold_positive(self):
        assert ProcessingConfig.DESPIKE["z_threshold"] > 0

    def test_rotation_max_angle_positive(self):
        assert ProcessingConfig.ROTATION["max_rotation_angle"] > 0


# ---------------------------------------------------------------------------
# ErrorCode
# ---------------------------------------------------------------------------

class TestErrorCode:
    def test_success_is_zero(self):
        assert int(ErrorCode.SUCCESS) == 0

    def test_all_codes_non_negative(self):
        for code in ErrorCode:
            assert int(code) >= 0


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

class TestUnitConversion:
    def test_ms_to_kmh(self):
        assert UNIT_CONVERSION["ms_to_kmh"] == pytest.approx(3.6)

    def test_pa_to_kpa(self):
        assert UNIT_CONVERSION["pa_to_kpa"] == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# get_displacement_height
# ---------------------------------------------------------------------------

class TestGetDisplacementHeight:
    def test_forest_canopy_20m(self):
        d = get_displacement_height(SurfaceType.FOREST, 20.0)
        expected = DISPLACEMENT_RATIO[SurfaceType.FOREST] * 20.0
        assert d == pytest.approx(expected)

    def test_bareland_is_zero(self):
        d = get_displacement_height(SurfaceType.BARELAND, 0.5)
        assert d == pytest.approx(0.0)

    def test_zero_canopy_returns_zero(self):
        assert get_displacement_height(SurfaceType.CROP, 0.0) == 0.0

    def test_negative_canopy_raises(self):
        with pytest.raises(ValueError):
            get_displacement_height(SurfaceType.GRASS, -1.0)

    def test_result_non_negative(self):
        for st in SurfaceType:
            assert get_displacement_height(st, 10.0) >= 0.0


# ---------------------------------------------------------------------------
# get_roughness_length
# ---------------------------------------------------------------------------

class TestGetRoughnessLength:
    def test_crop_uses_empirical_factor(self):
        z0 = get_roughness_length(SurfaceType.CROP, canopy_height=2.0)
        assert z0 == pytest.approx(0.15 * 2.0)

    def test_grass_uses_empirical_factor(self):
        z0 = get_roughness_length(SurfaceType.GRASS, canopy_height=0.5)
        assert z0 == pytest.approx(0.15 * 0.5)

    def test_forest_uses_table(self):
        z0 = get_roughness_length(SurfaceType.FOREST, canopy_height=20.0)
        assert z0 == pytest.approx(ROUGHNESS_LENGTH[SurfaceType.FOREST])

    def test_custom_value_overrides(self):
        z0 = get_roughness_length(SurfaceType.FOREST, canopy_height=20.0, custom_value=2.5)
        assert z0 == pytest.approx(2.5)

    def test_negative_canopy_raises(self):
        with pytest.raises(ValueError):
            get_roughness_length(SurfaceType.CROP, canopy_height=-0.1)

    def test_result_positive(self):
        for st in SurfaceType:
            z0 = get_roughness_length(st, canopy_height=5.0)
            assert z0 > 0
