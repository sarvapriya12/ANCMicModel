"""Audio capture / playback for the lab app and the Pi.

Design rules (each fixes a defect of the previous version):

* **A/B can never desync.** The two microphone channels travel together as one
  ``(2, N)`` frame. On the Pi both I2S mics arrive as one stereo ALSA stream, so
  pairing is by construction. For two separate devices (lab testing) frames are
  paired in arrival order under a lock, and when one device runs ahead the
  *oldest* frames of the longer side are dropped to resynchronise (counted).
* **No polling.** ``get_frames`` blocks on a queue instead of ``sleep(0.01)``
  loops (which also add up to 10-15 ms of jitter on Windows).
* **Overflows / underruns are counted, never silent.** No ``print`` runs in an
  audio callback.
* **Capture and playback clocks differ.** Output goes through a PI-controlled
  elastic resampler (``highspl.dsp.drift``) instead of dropping or inventing a
  block every few minutes.
"""

import contextlib
import json
import os
import queue
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf

from highspl.config import project_root
from highspl.dsp.drift import DriftCompensator


class AudioCapture:
    def __init__(
        self,
        mic_a_id,
        mic_b_id,
        output_id,
        blocksize=512,
        sr=48000,
        queue_blocks=8,
        latency="low",
        drift_compensation=True,
        record_path=None,
    ):
        self.mic_a_id = mic_a_id
        self.mic_b_id = mic_b_id
        self.output_id = output_id
        self.blocksize = blocksize
        self.sr = sr
        self.latency = latency
        self.shared_stereo = mic_a_id == mic_b_id

        self.running = False
        self._frames = queue.Queue(maxsize=queue_blocks)  # (2, N) float32 pairs
        self._pair_lock = threading.Lock()
        self._pend_a = []
        self._pend_b = []

        self.drift = (
            DriftCompensator(target_fill=2 * blocksize) if drift_compensation else None
        )
        self._out_fifo = queue.Queue(maxsize=queue_blocks)  # used if drift is off

        self.stream_a = None
        self.stream_b = None
        self.stream_out = None

        # counters (read from the UI / worker thread; written from callbacks)
        self.stats = {
            "input_overflows": 0,
            "output_underflows": 0,
            "queue_drops": 0,
            "resync_drops": 0,
            "blocks_in": 0,
            "device_gap": 0,
        }
        self._gap = False  # an overflow happened since the last delivered frame

        self.recording = False
        self.record_path = str(
            record_path or (project_root() / "artifacts" / "recordings")
        )
        self.rec_a, self.rec_b, self.rec_enh, self.rec_metrics = [], [], [], []

    # ----------------------------------------------------------- callbacks
    def _enqueue(self, pair):
        """Put a (2, N) frame in the worker queue; drop the oldest on overflow
        (as a whole pair, so channels stay aligned)."""
        try:
            self._frames.put_nowait(pair)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            self.stats["queue_drops"] += 1
            self._gap = True
            try:
                self._frames.put_nowait(pair)
            except queue.Full:
                pass
        self.stats["blocks_in"] += 1

    def _note_status(self, status):
        if status:
            if getattr(status, "input_overflow", False):
                self.stats["input_overflows"] += 1
                self._gap = True
            if getattr(status, "output_underflow", False):
                self.stats["output_underflows"] += 1

    def _stereo_cb(self, indata, frames, time_info, status):
        self._note_status(status)
        if not self.running:
            return
        a = indata[:, 0]
        b = indata[:, 1] if indata.shape[1] > 1 else np.zeros_like(a)
        self._enqueue(np.stack((a, b)).astype(np.float32, copy=True))

    def _mono_cb(self, indata, frames, time_info, status, side):
        self._note_status(status)
        if not self.running:
            return
        block = indata[:, 0].astype(np.float32, copy=True)
        with self._pair_lock:
            (self._pend_a if side == "a" else self._pend_b).append(block)
            # resynchronise if one device runs ahead of the other
            while len(self._pend_a) - len(self._pend_b) > 2:
                self._pend_a.pop(0)
                self.stats["resync_drops"] += 1
            while len(self._pend_b) - len(self._pend_a) > 2:
                self._pend_b.pop(0)
                self.stats["resync_drops"] += 1
            while self._pend_a and self._pend_b:
                a, b = self._pend_a.pop(0), self._pend_b.pop(0)
                n = min(a.size, b.size)
                self._enqueue(np.stack((a[:n], b[:n])))

    def _mono_cb_a(self, indata, frames, time_info, status):
        self._mono_cb(indata, frames, time_info, status, "a")

    def _mono_cb_b(self, indata, frames, time_info, status):
        self._mono_cb(indata, frames, time_info, status, "b")

    def _out_cb(self, outdata, frames, time_info, status):
        self._note_status(status)
        if not self.running:
            outdata.fill(0)
            return
        if self.drift is not None:
            outdata[:, 0] = self.drift.read(frames)
            return
        try:
            data = self._out_fifo.get_nowait()
            if len(data) == frames:
                outdata[:, 0] = data
            else:
                outdata.fill(0)
                self.stats["output_underflows"] += 1
        except queue.Empty:
            outdata.fill(0)
            self.stats["output_underflows"] += 1

    # ------------------------------------------------------------- devices
    @staticmethod
    def _validate_device(dev_id, direction="input"):
        """Check that a device ID is valid and has the required channels."""
        try:
            devs = sd.query_devices()
            if dev_id is None:
                return False, "No device selected (ID is None)"
            if dev_id < 0 or dev_id >= len(devs):
                return (
                    False,
                    f"Device ID {dev_id} is out of range (system has {len(devs)} devices)",
                )
            dev = devs[dev_id]
            key = (
                "max_input_channels" if direction == "input" else "max_output_channels"
            )
            if dev[key] < 1:
                return (
                    False,
                    f"Device [{dev_id}] '{dev['name']}' has no {direction} channels",
                )
            return True, f"Device [{dev_id}] '{dev['name']}' OK"
        except Exception as e:  # noqa: BLE001
            return False, f"Device query failed: {e}"

    def start(self):
        for key in self.stats:
            self.stats[key] = 0
        self._gap = False
        self._pend_a.clear()
        self._pend_b.clear()
        while not self._frames.empty():
            self._frames.get_nowait()
        if self.drift is not None:
            self.drift.reset()

        checks = [
            ("Mic A", self.mic_a_id, "input"),
            ("Mic B", self.mic_b_id, "input"),
            ("Output", self.output_id, "output"),
        ]
        for label, dev_id, direction in checks:
            ok, msg = self._validate_device(dev_id, direction)
            if not ok:
                raise RuntimeError(f"{label} device invalid: {msg}")

        try:
            self.running = True
            if self.shared_stereo:
                # Pi 5 I2S: both mics are the L/R slots of one ALSA capture stream.
                dev_info = sd.query_devices(self.mic_a_id)
                channels = min(2, dev_info["max_input_channels"])
                if channels < 2:
                    raise RuntimeError(
                        "Shared-stereo mode needs a 2-channel input device; "
                        f"'{dev_info['name']}' only has {channels}. Select two different "
                        "mono devices or the I2S stereo capture card."
                    )
                self.stream_a = sd.InputStream(
                    device=self.mic_a_id,
                    channels=2,
                    samplerate=self.sr,
                    blocksize=self.blocksize,
                    latency=self.latency,
                    dtype="float32",
                    callback=self._stereo_cb,
                )
                self.stream_a.start()
            else:
                self.stream_a = sd.InputStream(
                    device=self.mic_a_id,
                    channels=1,
                    samplerate=self.sr,
                    blocksize=self.blocksize,
                    latency=self.latency,
                    dtype="float32",
                    callback=self._mono_cb_a,
                )
                self.stream_b = sd.InputStream(
                    device=self.mic_b_id,
                    channels=1,
                    samplerate=self.sr,
                    blocksize=self.blocksize,
                    latency=self.latency,
                    dtype="float32",
                    callback=self._mono_cb_b,
                )
                self.stream_a.start()
                self.stream_b.start()

            self.stream_out = sd.OutputStream(
                device=self.output_id,
                channels=1,
                samplerate=self.sr,
                blocksize=self.blocksize,
                latency=self.latency,
                dtype="float32",
                callback=self._out_cb,
            )
            self.stream_out.start()
        except Exception:
            self.stop()
            raise

    def stop(self):
        self.running = False
        for name in ("stream_a", "stream_b", "stream_out"):
            stream = getattr(self, name)
            if stream is not None:
                # closing must not mask the real error that led here
                with contextlib.suppress(Exception):
                    stream.stop()
                    stream.close()
                setattr(self, name, None)
        # wake a worker blocked in get_frames()
        try:
            self._frames.put_nowait(None)
        except queue.Full:
            pass
        if self.recording:
            self.save_session()

    # ----------------------------------------------------------- data path
    def get_frames(self, timeout=0.1):
        """Block for the next paired frame. Returns ``(mic_a, mic_b, gap)`` or
        ``(None, None, False)`` on timeout / shutdown. ``gap`` is True if audio
        was lost (overflow / queue drop) since the previous frame."""
        try:
            pair = self._frames.get(timeout=timeout)
        except queue.Empty:
            return None, None, False
        if pair is None:
            return None, None, False
        gap, self._gap = self._gap, False
        return pair[0], pair[1], gap

    def push_output(self, enhanced_audio, metrics=None):
        block = np.asarray(enhanced_audio, dtype=np.float32)
        if self.drift is not None:
            self.drift.write(block)
            self.stats["output_underflows"] = max(
                self.stats["output_underflows"], self.drift.underruns
            )
        else:
            try:
                self._out_fifo.put_nowait(block)
            except queue.Full:
                self.stats["queue_drops"] += 1
        if self.recording and metrics:
            self.rec_enh.append(block.copy())
            self.rec_a.append(
                np.asarray(metrics.get("waveform", np.zeros_like(block))).copy()
            )
            self.rec_b.append(
                np.asarray(metrics.get("waveform_ref", np.zeros_like(block))).copy()
            )
            self.rec_metrics.append(
                {
                    "noise_reduction_db": float(metrics.get("noise_reduction_db", 0)),
                    "coherence": float(metrics.get("coherence", 0)),
                    "latency": metrics.get("latency", {}),
                }
            )

    # ----------------------------------------------------------- recording
    def start_recording(self):
        self.recording = True
        self.rec_a, self.rec_b, self.rec_enh, self.rec_metrics = [], [], [], []

    def save_session(self):
        self.recording = False
        if not self.rec_enh:
            return None
        try:
            os.makedirs(self.record_path, exist_ok=True)
            a_data = np.concatenate(self.rec_a)
            b_data = np.concatenate(self.rec_b)
            enh_data = np.concatenate(self.rec_enh)
            sf.write(os.path.join(self.record_path, "mic_a.wav"), a_data, self.sr)
            sf.write(os.path.join(self.record_path, "mic_b.wav"), b_data, self.sr)
            sf.write(os.path.join(self.record_path, "enhanced.wav"), enh_data, self.sr)
            with open(os.path.join(self.record_path, "metrics.json"), "w") as f:
                json.dump(self.rec_metrics, f, indent=2)
            return self.record_path
        except Exception as e:  # noqa: BLE001
            self.stats["save_error"] = str(e)
            return None
