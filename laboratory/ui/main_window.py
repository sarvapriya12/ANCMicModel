import collections
import os
import sys
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
import soundfile as sf
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# Local imports: the repo root goes FIRST on sys.path and the lab packages are
# imported by their full names, so a site-packages "audio"/"dsp" can't shadow them.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from highspl.config import project_root
from highspl.telemetry import NoiseFloorTracker, latency_budget
from laboratory.audio.capture import AudioCapture
from laboratory.dsp.pipeline import HighSPLPipeline
from laboratory.visualization.plots import (
    CoherencePlot,
    DualWaveformPlot,
    LatencyBarChart,
    SpectrumPlot,
)

NN_LATENCY_SAMPLES = 512  # neural-stage algorithmic delay, measured by the evaluator


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HighSPL Dual Mic Speech Enhancement Laboratory")
        self.setGeometry(50, 50, 1500, 950)

        self.audio_capture = None
        self.dsp_pipeline = HighSPLPipeline()
        self.is_running = False
        self.worker_thread = None
        self._stop_event = threading.Event()
        self._pending_mode = None  # applied by the worker between blocks
        self.noise_in = NoiseFloorTracker()
        self.noise_out = NoiseFloorTracker()
        self._health = {"healthy": True, "clip": 0.0, "frozen": 0.0, "p99_ms": 0.0}

        # Thread-safe metrics ring buffer (audio thread pushes, UI timer polls)
        self.metrics_buffer = collections.deque(maxlen=10)

        # 30 FPS UI refresh timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._poll_metrics)

        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1a1a2e; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
            QGroupBox { border: 1px solid #3a3a5c; border-radius: 6px; margin-top: 14px; font-weight: bold; font-size: 13px; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #7ec8e3; }
            QPushButton { background-color: #0e639c; color: white; padding: 10px 16px; border: none; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #1177bb; }
            QPushButton:disabled { background-color: #444; }
            QComboBox { background-color: #2a2a4a; color: white; border: 1px solid #555; padding: 6px; border-radius: 3px; }
            QProgressBar { text-align: center; border: 1px solid #3a3a5c; background: #1a1a2e; border-radius: 3px; height: 14px; }
            QProgressBar::chunk { background-color: #4CAF50; border-radius: 3px; }
            QLabel { font-size: 12px; }
        """)

        self._build_ui()
        self._populate_devices()
        self._sync_mode_combo()

    def _build_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(10, 10, 10, 10)

        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(top_splitter, stretch=2)

        # ═══════════════════════════════════════
        # LEFT PANEL: MIC HARDWARE
        # ═══════════════════════════════════════
        left_panel = QGroupBox("MICROPHONE HARDWARE")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(6)

        # Mic A
        left_layout.addWidget(QLabel("PRIMARY MIC A (Voice + Noise)"))
        self.cb_mic_a = QComboBox()
        left_layout.addWidget(self.cb_mic_a)

        h_a = QHBoxLayout()
        h_a.addWidget(QLabel("Level:"))
        self.meter_a = QProgressBar()
        self.meter_a.setRange(0, 100)
        h_a.addWidget(self.meter_a)
        left_layout.addLayout(h_a)

        self.btn_test_a = QPushButton("TEST MIC A")
        self.btn_test_a.clicked.connect(lambda: self.test_mic("A"))
        left_layout.addWidget(self.btn_test_a)

        left_layout.addSpacing(16)

        # Mic B
        left_layout.addWidget(QLabel("REFERENCE MIC B (Noise Only)"))
        self.cb_mic_b = QComboBox()
        left_layout.addWidget(self.cb_mic_b)

        h_b = QHBoxLayout()
        h_b.addWidget(QLabel("Level:"))
        self.meter_b = QProgressBar()
        self.meter_b.setRange(0, 100)
        h_b.addWidget(self.meter_b)
        left_layout.addLayout(h_b)

        self.btn_test_b = QPushButton("TEST MIC B")
        self.btn_test_b.clicked.connect(lambda: self.test_mic("B"))
        left_layout.addWidget(self.btn_test_b)

        left_layout.addSpacing(10)

        self.btn_refresh = QPushButton("🔄 REFRESH DEVICES")
        self.btn_refresh.setStyleSheet(
            "background-color: #555; font-size: 11px; padding: 6px;"
        )
        self.btn_refresh.clicked.connect(self._refresh_devices)
        left_layout.addWidget(self.btn_refresh)

        left_layout.addStretch()

        top_splitter.addWidget(left_panel)

        # ═══════════════════════════════════════
        # MIDDLE PANEL: CONTROL & PIPELINE
        # ═══════════════════════════════════════
        middle_panel = QGroupBox("PIPELINE CONTROL & ENVIRONMENT")
        middle_layout = QVBoxLayout(middle_panel)
        middle_layout.setSpacing(8)

        # Status indicator
        status_row = QHBoxLayout()
        self.lbl_status = QLabel("OFFLINE")
        self.lbl_status.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #ff4444; padding: 4px;"
        )
        status_row.addWidget(self.lbl_status)
        middle_layout.addLayout(status_row)

        # Pipeline visual flow
        self.lbl_pipeline = QLabel(
            "Mic A + Mic B  -->  Trust  -->  NLMS  -->  AI  -->  Output"
        )
        self.lbl_pipeline.setStyleSheet(
            "font-size: 11px; color: #7ec8e3; padding: 6px; border: 1px dashed #3a3a5c; border-radius: 4px;"
        )
        self.lbl_pipeline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        middle_layout.addWidget(self.lbl_pipeline)

        middle_layout.addSpacing(10)

        # Noise Source
        middle_layout.addWidget(QLabel("Noise Source Simulation"))
        self.cb_noise = QComboBox()
        self.cb_noise.addItems(["None", "Custom WAV..."])
        self.cb_noise.currentTextChanged.connect(self.on_noise_changed)
        middle_layout.addWidget(self.cb_noise)

        middle_layout.addSpacing(10)

        # DSP Mode selector
        dsp_row = QHBoxLayout()
        dsp_row.addWidget(QLabel("Pre-Filter:"))
        self.cb_dsp_mode = QComboBox()
        self.cb_dsp_mode.addItems(
            [
                "ROBUST (Hard Gated)",
                "VSS (Academic)",
                "CLASSICAL (Baseline)",
                "FDAF (Battlefield)",
                "DIFF (Differential)",
                "ADIFF (Adaptive Differential)",
                "GSC (Beamformer + Canceller)",
                "BYPASS (AI Only)",
            ]
        )
        self.cb_dsp_mode.setCurrentText("ROBUST (Hard Gated)")
        self.cb_dsp_mode.currentTextChanged.connect(self.on_dsp_changed)
        dsp_row.addWidget(self.cb_dsp_mode)
        middle_layout.addLayout(dsp_row)

        # Blocksize selector
        bs_row = QHBoxLayout()
        bs_row.addWidget(QLabel("Block Size:"))
        self.cb_blocksize = QComboBox()
        self.cb_blocksize.addItems(["512 (AI Locked)"])
        self.cb_blocksize.setCurrentText("512 (AI Locked)")
        self.cb_blocksize.setEnabled(False)
        bs_row.addWidget(self.cb_blocksize)
        middle_layout.addLayout(bs_row)

        middle_layout.addSpacing(10)

        # Enhancement toggle
        self.chk_enhance = QCheckBox("ENABLE HIGH-SPL ENHANCEMENT")
        self.chk_enhance.setChecked(True)
        self.chk_enhance.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #4CAF50;"
        )
        middle_layout.addWidget(self.chk_enhance)

        middle_layout.addSpacing(10)

        # Start button
        self.btn_start = QPushButton("START REAL-TIME VALIDATION")
        self.btn_start.setStyleSheet(
            "background-color: #28a745; font-size: 14px; padding: 12px;"
        )
        self.btn_start.clicked.connect(self.toggle_processing)
        middle_layout.addWidget(self.btn_start)

        # Save session button
        self.btn_save = QPushButton("SAVE TEST SESSION")
        self.btn_save.setStyleSheet("background-color: #6f42c1;")
        self.btn_save.clicked.connect(self.trigger_save_session)
        middle_layout.addWidget(self.btn_save)

        middle_layout.addStretch()
        top_splitter.addWidget(middle_panel)

        # ═══════════════════════════════════════
        # RIGHT PANEL: LISTENER OUTPUT
        # ═══════════════════════════════════════
        right_panel = QGroupBox("LISTENER OUTPUT")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(8)

        right_layout.addWidget(QLabel("Output Hardware"))
        self.cb_out = QComboBox()
        right_layout.addWidget(self.cb_out)

        h_out = QHBoxLayout()
        h_out.addWidget(QLabel("Output Level:"))
        self.meter_out = QProgressBar()
        self.meter_out.setRange(0, 100)
        h_out.addWidget(self.meter_out)
        right_layout.addLayout(h_out)

        self.btn_test_out = QPushButton("TEST OUTPUT (live_noisy_clean.wav)")
        self.btn_test_out.setStyleSheet("background-color: #6f42c1;")
        self.btn_test_out.clicked.connect(self.test_output_audio)
        right_layout.addWidget(self.btn_test_out)

        right_layout.addSpacing(16)

        # Metrics Display
        self.lbl_nr = QLabel("Noise floor in pauses: -- dBFS")
        self.lbl_nr.setStyleSheet("font-size: 18px; color: #4CAF50; font-weight: bold;")
        right_layout.addWidget(self.lbl_nr)

        self.lbl_lat = QLabel("Latency (lower bound): -- ms")
        self.lbl_lat.setStyleSheet("font-size: 14px; color: #7ec8e3;")
        right_layout.addWidget(self.lbl_lat)

        self.lbl_lat_detail = QLabel("DSP: -- ms  |  AI: -- ms")
        self.lbl_lat_detail.setStyleSheet("font-size: 12px; color: #aaa;")
        right_layout.addWidget(self.lbl_lat_detail)

        self.lbl_health = QLabel("Health: --")
        self.lbl_health.setStyleSheet("font-size: 12px; color: #aaa;")
        self.lbl_health.setWordWrap(True)
        right_layout.addWidget(self.lbl_health)

        right_layout.addStretch()
        top_splitter.addWidget(right_panel)

        # ═══════════════════════════════════════
        # BOTTOM: 4 ANALYSIS GRAPHS
        # ═══════════════════════════════════════
        bottom_panel = QGroupBox("ANALYSIS GRAPHS")
        bottom_layout = QGridLayout(bottom_panel)
        bottom_layout.setSpacing(6)
        main_layout.addWidget(bottom_panel, stretch=3)

        self.plot_raw = DualWaveformPlot()
        self.plot_spec = SpectrumPlot()
        self.plot_coh = CoherencePlot()
        self.plot_lat = LatencyBarChart()

        bottom_layout.addWidget(self.plot_raw, 0, 0)
        bottom_layout.addWidget(self.plot_spec, 0, 1)
        bottom_layout.addWidget(self.plot_coh, 1, 0)
        bottom_layout.addWidget(self.plot_lat, 1, 1)

    def _sync_mode_combo(self):
        """Show the pipeline's real stage-1 mode (the preset decides the default)."""
        mode = self.dsp_pipeline.dsp_mode
        for i in range(self.cb_dsp_mode.count()):
            if self.cb_dsp_mode.itemText(i).split(" ")[0] == mode:
                self.cb_dsp_mode.setCurrentIndex(i)
                break
        self.setWindowTitle(
            f"HighSPL Dual Mic Speech Enhancement Laboratory - preset {self.dsp_pipeline.preset}"
        )

    def _populate_devices(self):
        self.cb_mic_a.clear()
        self.cb_mic_b.clear()
        self.cb_out.clear()

        devs = sd.query_devices()
        hostapis = sd.query_hostapis()
        mme_api_idx = next(
            (i for i, api in enumerate(hostapis) if api["name"] == "MME"), None
        )

        for idx, d in enumerate(devs):
            name = f"{d['name']} (ID: {idx})"
            if (
                sys.platform == "win32"
                and mme_api_idx is not None
                and d["hostapi"] != mme_api_idx
            ):
                continue
            if d["max_input_channels"] > 0:
                self.cb_mic_a.addItem(name, idx)
                self.cb_mic_b.addItem(name, idx)
            if d["max_output_channels"] > 0:
                self.cb_out.addItem(name, idx)

    def _refresh_devices(self):
        """Re-scan audio devices (e.g. after plugging in a headset)."""
        was_running = self.is_running
        if was_running:
            self.toggle_processing()  # stop first
        self._populate_devices()
        print("[AUDIO] Device list refreshed.")

    # ═══════════════════════════════════════
    # INTERACTIVE FEATURES
    # ═══════════════════════════════════════

    def test_mic(self, mic_name):
        cb = self.cb_mic_a if mic_name == "A" else self.cb_mic_b
        dev_id = cb.currentData()
        if dev_id is None:
            return

        msg = QMessageBox(self)
        msg.setWindowTitle(f"Testing Mic {mic_name}")
        msg.setText("Recording 2 seconds... Speak or make noise!")
        msg.setStandardButtons(QMessageBox.StandardButton.NoButton)
        msg.show()
        QApplication.processEvents()

        try:
            recording = sd.rec((48000 * 2), samplerate=48000, channels=1, device=dev_id)
            sd.wait()
            msg.accept()

            peak = np.max(np.abs(recording))
            rms = np.sqrt(np.mean(recording**2))
            rms_db = 20 * np.log10(rms + 1e-10)
            clipping = "YES - CLIPPING DETECTED" if peak > 0.95 else "No"
            # Noise floor = quietest 10 % of 20 ms windows (a muted mic is fine, not a crash)
            win = 960
            frames = recording[: (recording.size // win) * win].reshape(-1, win)
            floor_db = 10 * np.log10(
                np.percentile(np.mean(frames**2, axis=1), 10) + 1e-20
            )

            QMessageBox.information(
                self,
                f"Mic {mic_name} Test Results",
                f"Capture Successful!\n\n"
                f"Peak Amplitude: {peak:.4f}\n"
                f"RMS Level: {rms_db:.1f} dB\n"
                f"Clipping: {clipping}\n"
                f"Noise Floor: {floor_db:.1f} dBFS",
            )
        except Exception as e:  # noqa: BLE001
            msg.accept()
            QMessageBox.critical(self, "Error", f"Failed to test microphone:\n{e!s}")

    def test_output_audio(self):
        path = str(project_root() / "output_sample" / "live_noisy_clean.wav")
        if not os.path.exists(path):
            QMessageBox.warning(self, "File Not Found", f"Could not find:\n{path}")
            return
        out_id = self.cb_out.currentData()
        try:
            data, fs = sf.read(path)
            sd.play(data, fs, device=out_id)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Error", f"Failed to play:\n{e!s}")

    def on_noise_changed(self, text):
        sd.stop()
        if text == "Custom WAV...":
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Noise WAV", str(project_root()), "Audio (*.wav)"
            )
            if path:
                try:
                    data, fs = sf.read(path)
                    sd.play(data, fs, loop=True)
                    self.cb_noise.setItemText(1, f"Playing: {os.path.basename(path)}")
                except Exception as e:  # noqa: BLE001
                    QMessageBox.critical(self, "Error", f"Failed to load:\n{e!s}")
                    self.cb_noise.setCurrentIndex(0)
            else:
                self.cb_noise.setCurrentIndex(0)
        else:
            self.cb_noise.setItemText(1, "Custom WAV...")

    def on_dsp_changed(self, text):
        mode = text.split(" ")[0]  # "ROBUST", "VSS", "GSC", "BYPASS", ...
        if self.is_running:
            self._pending_mode = mode  # the worker switches between blocks
        else:
            self.dsp_pipeline.dsp_mode = mode

    def trigger_save_session(self):
        if not self.is_running:
            QMessageBox.warning(
                self,
                "Not Running",
                "Start the validation first before saving a session.",
            )
            return
        if self.audio_capture and not self.audio_capture.recording:
            self.audio_capture.start_recording()
            self.btn_save.setText("RECORDING... (Click to Stop & Save)")
            self.btn_save.setStyleSheet(
                "background-color: #ffc107; color: black; font-weight: bold;"
            )
        elif self.audio_capture and self.audio_capture.recording:
            self.audio_capture.save_session()
            self.btn_save.setText("SAVE TEST SESSION")
            self.btn_save.setStyleSheet("background-color: #6f42c1;")
            QMessageBox.information(
                self,
                "Session Saved",
                "Saved: mic_a.wav, mic_b.wav, enhanced.wav, metrics.json",
            )

    # ═══════════════════════════════════════
    # CORE PROCESSING ENGINE
    # ═══════════════════════════════════════

    def toggle_processing(self):
        if not self.is_running:
            id_a = self.cb_mic_a.currentData()
            id_b = self.cb_mic_b.currentData()
            id_out = self.cb_out.currentData()

            if id_a is None or id_b is None or id_out is None:
                QMessageBox.warning(
                    self,
                    "No Device Selected",
                    "Please select a valid Mic A, Mic B, and Output device.\n\n"
                    "If no devices appear, check that your microphone and speakers\n"
                    "are connected and recognized by Windows.",
                )
                return

            bs = int(self.cb_blocksize.currentText().split()[0])

            try:
                self.audio_capture = AudioCapture(id_a, id_b, id_out, blocksize=bs)
                self.audio_capture.start()
            except RuntimeError as e:
                QMessageBox.critical(
                    self,
                    "Audio Device Error",
                    f"Could not open audio streams:\n\n{e}\n\n"
                    "Try selecting different devices from the dropdowns,\n"
                    "or reconnect your headset and click the Refresh button.",
                )
                self.audio_capture = None
                return
            except Exception as e:  # noqa: BLE001
                QMessageBox.critical(
                    self, "Audio Error", f"Unexpected error starting audio:\n\n{e}"
                )
                self.audio_capture = None
                return

            # Every session starts from clean state (stale NLMS weights / model
            # caches from a previous device selection must not leak in).
            self.dsp_pipeline.reset()
            self.noise_in.reset()
            self.noise_out.reset()
            self._stop_event.clear()
            self.is_running = True
            self.lbl_status.setText("RUNNING")
            self.lbl_status.setStyleSheet(
                "font-size: 16px; font-weight: bold; color: #4CAF50; padding: 4px;"
            )
            self.btn_start.setText("STOP VALIDATION")
            self.btn_start.setStyleSheet(
                "background-color: #dc3545; font-size: 14px; padding: 12px;"
            )

            # Start 30 FPS UI polling
            self.update_timer.start(33)

            self.worker_thread = threading.Thread(
                target=self._processing_worker, daemon=True
            )
            self.worker_thread.start()
        else:
            self.is_running = False
            self._stop_event.set()
            self.update_timer.stop()
            if self.audio_capture:
                self.audio_capture.stop()  # also wakes a worker blocked on the queue
            if self.worker_thread is not None:
                self.worker_thread.join(
                    timeout=1.0
                )  # never two workers on one pipeline
                self.worker_thread = None
            self.lbl_status.setText("OFFLINE")
            self.lbl_status.setStyleSheet(
                "font-size: 16px; font-weight: bold; color: #ff4444; padding: 4px;"
            )
            self.btn_start.setText("START REAL-TIME VALIDATION")
            self.btn_start.setStyleSheet(
                "background-color: #28a745; font-size: 14px; padding: 12px;"
            )

    def _processing_worker(self):
        """Runs in a background thread. Never touches the UI directly and never
        dies silently: any error is contained, counted and shown as ``Health``."""
        last_log = time.time()
        block_ms = deque(maxlen=400)
        frozen = deque(maxlen=200)
        clipped = deque(maxlen=200)

        while self.is_running and not self._stop_event.is_set():
            capture = self.audio_capture
            if capture is None:
                break
            mic_a, mic_b, gap = capture.get_frames(timeout=0.1)
            if mic_a is None:
                continue
            if self._pending_mode is not None:
                self.dsp_pipeline.dsp_mode = self._pending_mode
                self._pending_mode = None

            try:
                if self.chk_enhance.isChecked():
                    t0 = time.perf_counter()
                    metrics = self.dsp_pipeline.process(mic_a, mic_b)
                    block_ms.append((time.perf_counter() - t0) * 1000.0)
                    tele = metrics["telemetry"]
                    frozen.append(1.0 if tele.get("adaptation_frozen") else 0.0)
                    clipped.append(tele.get("clip_front", 0.0))
                    capture.push_output(metrics["enhanced_audio"], metrics)
                else:
                    metrics = {
                        "waveform": mic_a,
                        "waveform_ref": mic_b,
                        "enhanced_audio": mic_a,
                        "noise_reduction_db": 0.0,
                        "coherence": 0.0,
                        "latency": {"dsp": 0.0, "ai": 0.0},
                        "telemetry": {"healthy": True},
                    }
                    capture.push_output(mic_a)
                self.noise_in.update(mic_a)
                self.noise_out.update(metrics["enhanced_audio"])
                self._health = {
                    "healthy": bool(metrics["telemetry"].get("healthy", True))
                    and not gap,
                    "clip": float(np.mean(clipped)) if clipped else 0.0,
                    "frozen": float(np.mean(frozen)) if frozen else 0.0,
                    "p99_ms": float(np.percentile(block_ms, 99)) if block_ms else 0.0,
                    "xruns": dict(capture.stats),
                }
                self.metrics_buffer.append(metrics)
            except Exception as e:  # noqa: BLE001 - keep the audio thread alive
                self._health = {**self._health, "healthy": False, "error": str(e)}
                continue

            now = time.time()
            if now - last_log > 1.0:
                va = float(np.max(np.abs(mic_a)))
                vb = float(np.max(np.abs(mic_b)))
                lat = metrics["latency"]
                print(
                    f"[TELEMETRY] A:{va:.4f} B:{vb:.4f} "
                    f"floor_in:{self.noise_in.floor_dbfs:.1f} "
                    f"floor_out:{self.noise_out.floor_dbfs:.1f} "
                    f"DSP:{lat['dsp']:.1f}ms AI:{lat['ai']:.1f}ms "
                    f"xruns:{capture.stats['input_overflows']}",
                    flush=True,
                )
                last_log = now

    def _poll_metrics(self):
        """Called by QTimer at 30 FPS. Pulls latest metrics and updates UI."""
        try:
            metrics = self.metrics_buffer.pop()
            self.metrics_buffer.clear()  # Drop stale frames
            self._update_graphs(metrics)
        except IndexError:
            pass

    def _update_graphs(self, metrics):
        # Meters
        va = (
            np.clip(np.max(np.abs(metrics["waveform"])) * 100, 0, 100)
            if len(metrics["waveform"]) > 0
            else 0
        )
        vb = (
            np.clip(np.max(np.abs(metrics["waveform_ref"])) * 100, 0, 100)
            if len(metrics["waveform_ref"]) > 0
            else 0
        )
        vo = (
            np.clip(np.max(np.abs(metrics["enhanced_audio"])) * 100, 0, 100)
            if len(metrics["enhanced_audio"]) > 0
            else 0
        )
        self.meter_a.setValue(int(va))
        self.meter_b.setValue(int(vb))
        self.meter_out.setValue(int(vo))

        # Text metrics (honest versions)
        dsp_lat = metrics["latency"]["dsp"]
        ai_lat = metrics["latency"]["ai"]
        f_in, f_out = self.noise_in.floor_dbfs, self.noise_out.floor_dbfs
        if np.isfinite(f_in) and np.isfinite(f_out):
            self.lbl_nr.setText(
                f"Noise floor: {f_in:.0f} -> {f_out:.0f} dBFS ({f_out - f_in:+.0f} dB)"
            )
        bs = self.audio_capture.blocksize if self.audio_capture else 512
        budget = latency_budget(
            bs,
            model_latency_samples=NN_LATENCY_SAMPLES,
            compute_ms=dsp_lat + ai_lat,
        )
        self.lbl_lat.setText(
            f"Latency (lower bound): {budget['total_lower_bound_ms']:.1f} ms"
        )
        self.lbl_lat_detail.setText(
            f"capture {budget['capture_block_ms']:.1f} + model {budget['model_ms']:.1f} + "
            f"DSP {dsp_lat:.1f} + AI {ai_lat:.1f} + "
            f"playback {budget['playback_block_ms']:.1f} ms (excl. OS/driver)"
        )
        h = self._health
        x = h.get("xruns", {})
        flag = "OK" if h.get("healthy", True) else "ATTENTION"
        self.lbl_health.setText(
            f"Health: {flag}  |  xruns in/out: {x.get('input_overflows', 0)}/"
            f"{x.get('output_underflows', 0)}  drops: {x.get('queue_drops', 0)}  |  "
            f"clip: {h.get('clip', 0) * 100:.1f}%  frozen: {h.get('frozen', 0) * 100:.0f}%  "
            f"p99 block: {h.get('p99_ms', 0):.1f} ms"
            + (f"  | error: {h['error']}" if "error" in h else "")
        )

        # Graphs
        self.plot_raw.update_data(metrics["waveform"], metrics["waveform_ref"])
        self.plot_spec.update_data(metrics["waveform"], metrics["enhanced_audio"])
        self.plot_coh.update_data(metrics["waveform"], metrics["waveform_ref"])
        self.plot_lat.update_data(dsp_lat, ai_lat)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
