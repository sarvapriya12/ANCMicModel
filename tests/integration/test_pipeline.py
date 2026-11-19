import numpy as np
import pytest

from highspl.config import load_config, project_root

ONNX = (
    project_root() / "models" / "fastenhancer" / "b" / "fastenhancer_b_streaming.onnx"
)
pytestmark = pytest.mark.skipif(
    not ONNX.is_file(),
    reason="FastEnhancer ONNX model not available (excluded from CI)",
)

PRESETS = sorted(
    p.stem for p in (project_root() / "configs" / "pipelines").glob("*.yaml")
)
B = 512


def _pipe(name):
    from highspl.evaluation.runner import build_pipeline

    return build_pipeline(name)


def _blocks(seed, n=12, level=0.05):
    rng = np.random.default_rng(seed)
    return [
        (
            (level * rng.standard_normal(B)).astype(np.float32),
            (level * rng.standard_normal(B)).astype(np.float32),
        )
        for _ in range(n)
    ]


@pytest.mark.parametrize("preset", PRESETS)
def test_every_preset_runs_and_is_finite(preset):
    from highspl.config import resolve_path
    from highspl.evaluation.runner import load_pipeline_config

    onnx = resolve_path(load_pipeline_config(preset).model.onnx_path)
    if not onnx.is_file():
        pytest.skip(f"model for preset {preset!r} not available: {onnx.name}")
    pipe = _pipe(preset)
    pipe.reset()
    for d, x in _blocks(0):
        res = pipe.process(d, x)
        assert res.enhanced.shape == (B,)
        assert np.isfinite(res.enhanced).all()
        assert res.telemetry["healthy"]


def test_audio_path_never_raises_on_nonfinite_input():
    pipe = _pipe("bypass")
    pipe.reset()
    d, x = _blocks(1, n=1)[0]
    d[10], x[20] = np.nan, np.inf

    res = pipe.process(d, x)

    assert np.isfinite(res.enhanced).all()
    assert res.telemetry["nonfinite_samples"] == 2


def test_failing_frontend_is_contained_and_reported():
    class Broken:
        name = "broken"

        def reset(self):
            pass

        def process(self, primary, reference):
            raise RuntimeError("boom")

        def adaptation_frozen(self):
            return False

    from highspl.pipeline import StreamingPipeline

    pipe = StreamingPipeline(load_config(), frontend=Broken())
    d, x = _blocks(2, n=1)[0]
    res = pipe.process(d, x)

    assert np.isfinite(res.enhanced).all()
    assert res.telemetry["healthy"] is False and res.telemetry["frontend_reset"]


def test_reset_makes_a_stream_reproducible():
    pipe = _pipe("gsc_blend")
    blocks = _blocks(3, n=20, level=0.1)

    def run():
        pipe.reset()
        return np.concatenate([pipe.process(d, x).enhanced for d, x in blocks])

    np.testing.assert_allclose(run(), run(), rtol=1e-5, atol=1e-6)


def test_set_mode_starts_from_clean_state():
    pipe = _pipe("B0")
    for d, x in _blocks(4, n=30, level=0.1):
        pipe.process(d, x)
    pipe.set_mode("gsc")

    assert pipe.mode == "gsc"
    fresh = _pipe("B0")
    fresh.set_mode("gsc")
    d, x = _blocks(5, n=1)[0]
    np.testing.assert_allclose(
        pipe.frontend.process(d, x), fresh.frontend.process(d, x), atol=1e-6
    )


def test_limiter_ceiling_holds_through_the_whole_pipeline():
    pipe = _pipe("bypass_agc")  # limiter at -1 dBFS
    pipe.reset()
    rng = np.random.default_rng(6)
    peak = 0.0
    for _ in range(40):
        d = (0.9 * rng.standard_normal(B)).astype(np.float32)
        peak = max(peak, float(np.abs(pipe.process(d, d.copy()).enhanced).max()))

    assert peak <= 10 ** (-1.0 / 20) + 1e-3


def test_measured_latency_matches_the_documented_budget():
    from highspl.evaluation.runner import calibrate_latency

    assert calibrate_latency(_pipe("bypass")) == pytest.approx(512, abs=4)
    # beamformer alignment adds only a few tenths of a millisecond
    assert 512 <= calibrate_latency(_pipe("gsc")) <= 512 + 40


def test_stage1_gain_is_unity_without_agc_and_matches_it_with():
    pipe = _pipe("bypass")
    d, x = _blocks(7, n=1)[0]
    out, gain, *_ = pipe.stage1(d, x)
    assert np.all(gain == 1.0) and out.shape == (B,)

    pipe = _pipe("bypass_agc")
    pipe.reset()
    for d, x in _blocks(8, n=30, level=0.01):
        out, gain, d2, _x2, _t = pipe.stage1(d, x)
    np.testing.assert_allclose(out, d2 * gain, rtol=1e-5, atol=1e-7)
