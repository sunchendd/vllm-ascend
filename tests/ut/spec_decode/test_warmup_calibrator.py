from unittest.mock import patch

from tests.ut.base import TestBase
from vllm_ascend.spec_decode.warmup_calibrator import (CalibrationMetric,
                                                       ComparisonResult,
                                                       SmartWarmupCalibrator)


def _make_result(concurrency: int, winner: str, improvement: float,
                 error_rate: float = 0.0) -> ComparisonResult:
    direct_tps = 100.0
    spec_tps = direct_tps * (1.0 + improvement)
    direct_metric = CalibrationMetric(
        tps=direct_tps,
        latency=100.0,
        cv=0.01,
        samples=5,
        error_rate=error_rate,
    )
    spec_metric = CalibrationMetric(
        tps=spec_tps,
        latency=100.0,
        cv=0.01,
        samples=5,
        error_rate=error_rate,
    )
    return ComparisonResult(
        concurrency=concurrency,
        spec_metric=spec_metric,
        direct_metric=direct_metric,
        winner=winner,
        improvement=improvement,
    )


class TestWarmupCalibrator(TestBase):

    def test_max_tie_uses_conservative_threshold(self):
        calibrator = SmartWarmupCalibrator(max_concurrency=64)

        mapping = {
            1: _make_result(1, "spec", 0.20),
            2: _make_result(2, "spec", 0.20),
            4: _make_result(4, "spec", 0.20),
            8: _make_result(8, "spec", 0.15),
            16: _make_result(16, "spec", 0.10),
            32: _make_result(32, "spec", 0.05),
            64: _make_result(64, "tie", 0.01),
        }

        with patch.object(calibrator, "_wait_for_service", return_value=True), \
                patch.object(calibrator, "_warmup_service", return_value=None), \
                patch.object(calibrator, "_switch_mode", return_value=None), \
                patch.object(calibrator, "_compare", side_effect=lambda p: mapping[p]):
            result = calibrator.run_calibration()

        self.assertIsNotNone(result)
        self.assertEqual(result.optimal_threshold, 32)

    def test_max_reliable_spec_win_keeps_max_threshold(self):
        calibrator = SmartWarmupCalibrator(max_concurrency=64)

        mapping = {
            1: _make_result(1, "spec", 0.20),
            2: _make_result(2, "spec", 0.20),
            4: _make_result(4, "spec", 0.20),
            8: _make_result(8, "spec", 0.15),
            16: _make_result(16, "spec", 0.10),
            32: _make_result(32, "spec", 0.08),
            64: _make_result(64, "spec", 0.06),
        }

        with patch.object(calibrator, "_wait_for_service", return_value=True), \
                patch.object(calibrator, "_warmup_service", return_value=None), \
                patch.object(calibrator, "_switch_mode", return_value=None), \
                patch.object(calibrator, "_compare", side_effect=lambda p: mapping[p]):
            result = calibrator.run_calibration()

        self.assertIsNotNone(result)
        self.assertEqual(result.optimal_threshold, 64)

    def test_binary_search_tie_is_conservative(self):
        calibrator = SmartWarmupCalibrator(max_concurrency=16)

        mapping = {
            1: _make_result(1, "spec", 0.20),
            2: _make_result(2, "spec", 0.20),
            4: _make_result(4, "spec", 0.15),
            8: _make_result(8, "spec", 0.08),
            16: _make_result(16, "direct", -0.12),
            12: _make_result(12, "tie", 0.01),
            10: _make_result(10, "spec", 0.05),
            11: _make_result(11, "tie", 0.00),
        }

        with patch.object(calibrator, "_wait_for_service", return_value=True), \
                patch.object(calibrator, "_warmup_service", return_value=None), \
                patch.object(calibrator, "_switch_mode", return_value=None), \
                patch.object(calibrator, "_compare", side_effect=lambda p: mapping[p]):
            result = calibrator.run_calibration()

        self.assertIsNotNone(result)
        self.assertEqual(result.optimal_threshold, 10)
