import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from laboratory.audio.capture import AudioCapture

B = 512


def _cap(**kw):
    cap = AudioCapture(1, 1, 2, blocksize=B, **kw)
    cap.running = True
    return cap


def _frame(seed, n=B):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


def test_stereo_frames_stay_paired_and_ordered():
    cap = _cap()
    sent = []
    for i in range(5):
        a, b = _frame(i), _frame(100 + i)
        sent.append((a, b))
        cap._stereo_cb(np.stack((a, b), axis=1), B, None, None)

    for a, b in sent:
        ga, gb, gap = cap.get_frames(timeout=0.1)
        np.testing.assert_array_equal(ga, a)
        np.testing.assert_array_equal(gb, b)
        assert gap is False


def test_a_reader_racing_the_callback_can_never_split_a_pair():
    """The old two-queue design lost A frames when the worker popped between the
    callback's two appends. A pair travels as one object now."""
    cap = _cap(queue_blocks=64)
    n = 400
    got = []
    stop = threading.Event()

    def reader():
        while not stop.is_set() or not cap._frames.empty():
            a, b, _ = cap.get_frames(timeout=0.01)
            if a is not None:
                got.append((a[0], b[0]))

    t = threading.Thread(target=reader)
    t.start()
    for i in range(n):
        a = np.full(B, i, dtype=np.float32)
        b = np.full(B, -i, dtype=np.float32)
        cap._stereo_cb(np.stack((a, b), axis=1), B, None, None)
        if i % 7 == 0:
            time.sleep(0.0005)
    stop.set()
    t.join()

    # Every frame is either delivered or counted as an overflow drop (a stress
    # reader may fall behind), and a delivered pair is NEVER split or reordered.
    assert len(got) + cap.stats["queue_drops"] == n
    assert all(a == -b for a, b in got)  # channel A and B of the same instant
    firsts = [int(a) for a, _ in got]
    assert firsts == sorted(firsts) and len(set(firsts)) == len(firsts)


def test_two_device_mode_pairs_in_arrival_order_and_resyncs():
    cap = AudioCapture(1, 2, 3, blocksize=B)
    cap.running = True
    # A delivers 5 blocks before B delivers any: A runs ahead
    for i in range(5):
        cap._mono_cb_a(_frame(i)[:, None], B, None, None)
    assert cap.stats["resync_drops"] == 3  # kept only the newest 2

    for i in range(2):
        cap._mono_cb_b(_frame(50 + i)[:, None], B, None, None)
    a, b, _ = cap.get_frames(timeout=0.1)
    np.testing.assert_array_equal(
        a, _frame(3)
    )  # oldest *kept* A block (not the stale ones)
    np.testing.assert_array_equal(b, _frame(50))


def test_queue_overflow_drops_whole_pairs_and_flags_a_gap():
    cap = _cap(queue_blocks=3)
    for i in range(6):
        cap._stereo_cb(
            np.stack((np.full(B, i), np.full(B, -i)), axis=1).astype(np.float32),
            B,
            None,
            None,
        )

    assert cap.stats["queue_drops"] == 3
    seen = []
    for _ in range(3):
        a, b, gap = cap.get_frames(timeout=0.1)
        assert a[0] == -b[0]
        seen.append((int(a[0]), gap))
    assert [v for v, _ in seen] == [3, 4, 5]  # newest kept, oldest dropped
    assert seen[0][1] is True  # first frame after the loss is flagged


def test_status_flags_are_counted_not_printed(capsys):
    cap = _cap()
    in_status = SimpleNamespace(input_overflow=True, output_underflow=False)
    out_status = SimpleNamespace(input_overflow=False, output_underflow=True)
    cap._stereo_cb(np.zeros((B, 2), dtype=np.float32), B, None, in_status)
    cap._out_cb(np.zeros((B, 1), dtype=np.float32), B, None, out_status)

    assert cap.stats["input_overflows"] == 1
    assert cap.stats["output_underflows"] >= 1
    assert capsys.readouterr().out == ""  # no I/O in audio callbacks


def test_get_frames_times_out_and_stop_wakes_a_blocked_worker():
    cap = _cap()
    assert cap.get_frames(timeout=0.01) == (None, None, False)

    result = {}

    def worker():
        result["r"] = cap.get_frames(timeout=5.0)

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.05)
    t0 = time.perf_counter()
    cap.stop()
    t.join(timeout=2.0)

    assert not t.is_alive()
    assert time.perf_counter() - t0 < 1.0
    assert result["r"] == (None, None, False)


def test_output_goes_through_the_drift_compensator():
    cap = _cap()
    cap.drift.write(np.zeros(2 * B, dtype=np.float32))  # prime to the target fill
    tone = np.sin(2 * np.pi * 440 * np.arange(B) / 48000).astype(np.float32)
    cap.push_output(tone)
    out = np.zeros((B, 1), dtype=np.float32)
    for _ in range(3):  # two blocks of priming silence, then the tone
        cap._out_cb(out, B, None, None)

    assert np.isfinite(out).all()
    assert np.abs(out).max() > 0.5  # the tone came out of the elastic FIFO
    assert cap.drift.underruns == 0


def test_mono_fallback_device_gives_zero_reference_instead_of_crashing():
    cap = _cap()
    cap._stereo_cb(np.ones((B, 1), dtype=np.float32), B, None, None)
    a, b, _ = cap.get_frames(timeout=0.1)

    assert a.min() == 1.0 and b.max() == 0.0


def test_save_session_writes_wavs_to_a_configurable_dir(tmp_path):
    cap = AudioCapture(1, 1, 2, blocksize=B, record_path=tmp_path)
    cap.running = True
    cap.start_recording()
    cap.push_output(_frame(1), {"waveform": _frame(2), "waveform_ref": _frame(3)})
    saved = cap.save_session()

    assert saved == str(tmp_path)
    assert (tmp_path / "enhanced.wav").is_file()
    assert (tmp_path / "metrics.json").is_file()


def test_start_rejects_invalid_devices_without_opening_streams():
    cap = AudioCapture(10_000, 10_000, 10_000, blocksize=B)
    with pytest.raises(RuntimeError, match="invalid"):
        cap.start()
    assert cap.stream_a is None and not cap.running
