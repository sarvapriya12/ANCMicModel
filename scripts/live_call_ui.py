"""Live dual-microphone call UI (customtkinter).

Runs the SAME streaming pipeline and capture code as the PyQt lab app
(`highspl.pipeline.StreamingPipeline`, `laboratory.audio.capture.AudioCapture`),
so mic pairing, xrun accounting, clock-drift compensation, fault containment and
the neural stage are identical. Preset: ``--preset`` (default ``gsc_blend``).
"""

import os
import sys
import threading

import customtkinter as ctk
import numpy as np
import sounddevice as sd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from highspl.evaluation.runner import build_pipeline, default_preset
from laboratory.audio.capture import AudioCapture

# --- UI Theme Setup ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

PRESET = default_preset()


class AudioStreamer:
    """Capture -> StreamingPipeline -> playback, with a contained worker thread."""

    def __init__(self, preset: str = PRESET):
        self.pipeline = build_pipeline(preset)
        self.capture = None
        self.thread = None
        self._stop = threading.Event()
        self.running = False
        self.vol_callback = None
        self.last_error = None

    def start_stream(self, mic_a_id, mic_b_id, output_id, enable_output):
        if self.running:
            return True, "Streaming Active"
        try:
            self.capture = AudioCapture(mic_a_id, mic_b_id, output_id)
            self.capture.start()
        except Exception as e:  # noqa: BLE001 - report to the UI instead of crashing
            self.capture = None
            return False, str(e)

        self.pipeline.reset()  # never start from a previous session's state
        self._stop.clear()
        self.last_error = None
        self.running = True
        self.thread = threading.Thread(
            target=self._loop, args=(enable_output,), daemon=True
        )
        self.thread.start()
        return True, "Streaming Active"

    def stop_stream(self):
        self.running = False
        self._stop.set()
        if self.capture is not None:
            self.capture.stop()  # also wakes a worker blocked on the queue
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        self.capture = None

    def _loop(self, enable_output):
        capture = self.capture
        while not self._stop.is_set():
            mic_a, mic_b, _gap = capture.get_frames(timeout=0.1)
            if mic_a is None:
                continue
            try:
                res = self.pipeline.process(mic_a, mic_b)
                if enable_output:
                    capture.push_output(res.enhanced)
                if self.vol_callback:
                    v_in = float(np.clip(np.max(np.abs(mic_a)), 0, 1))
                    v_out = float(np.clip(np.max(np.abs(res.enhanced)), 0, 1))
                    self.vol_callback(v_in, v_out)
            except Exception as e:  # noqa: BLE001 - keep the audio thread alive
                self.last_error = str(e)


class ModernLiveUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("High-SPL Dual Mic Commander")
        self.geometry("900x550")
        self.streamer = AudioStreamer()
        self.streamer.vol_callback = self.update_meters

        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.pack(side="left", fill="y")

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="SIH High-SPL SE",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        self.logo_label.pack(pady=(30, 10), padx=20)

        self.info_label = ctk.CTkLabel(
            self.sidebar_frame,
            text=f"Dual-mic front-end + neural enhancer\npreset: {PRESET}",
            font=ctk.CTkFont(size=14),
            text_color="gray",
        )
        self.info_label.pack(pady=10)

        # --- Main Content ---
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        # Devices
        self.in_devices = self.get_devices("input")
        self.out_devices = self.get_devices("output")
        dev_list = list(self.in_devices.keys())

        # MIC A
        ctk.CTkLabel(
            self.main_frame,
            text="Mic A (Primary: Voice + Noise)",
            font=ctk.CTkFont(weight="bold", size=14),
        ).pack(anchor="w", pady=(10, 0), padx=20)
        self.combo_a = ctk.CTkComboBox(self.main_frame, values=dev_list, width=450)
        self.combo_a.pack(anchor="w", pady=(2, 10), padx=20)

        # MIC B
        ctk.CTkLabel(
            self.main_frame,
            text="Mic B (Reference: Extreme Noise)",
            font=ctk.CTkFont(weight="bold", size=14),
        ).pack(anchor="w", padx=20)
        self.combo_b = ctk.CTkComboBox(self.main_frame, values=dev_list, width=450)
        self.combo_b.pack(anchor="w", pady=(2, 10), padx=20)

        # OUTPUT
        ctk.CTkLabel(
            self.main_frame,
            text="Output Hardware (Bluetooth/Speaker)",
            font=ctk.CTkFont(weight="bold", size=14),
        ).pack(anchor="w", padx=20)
        self.out_combo = ctk.CTkComboBox(
            self.main_frame, values=list(self.out_devices.keys()), width=450
        )
        self.out_combo.pack(anchor="w", pady=(2, 10), padx=20)

        # Output Toggle
        self.stream_output_var = ctk.BooleanVar(value=True)
        self.stream_switch = ctk.CTkSwitch(
            self.main_frame,
            text="Enable Real-Time Output Streaming",
            variable=self.stream_output_var,
            font=ctk.CTkFont(weight="bold"),
        )
        self.stream_switch.pack(anchor="w", pady=10, padx=20)

        # Controls
        self.btn_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.btn_frame.pack(fill="x", pady=10, padx=20)

        self.start_btn = ctk.CTkButton(
            self.btn_frame,
            text="Start Streaming",
            command=self.toggle_stream,
            fg_color="#28a745",
            hover_color="#218838",
            font=ctk.CTkFont(weight="bold", size=15),
            height=40,
        )
        self.start_btn.pack(side="left")

        self.status_label = ctk.CTkLabel(
            self.btn_frame,
            text="Status: IDLE",
            text_color="gray",
            font=ctk.CTkFont(size=14),
        )
        self.status_label.pack(side="left", padx=20)

        # Meters
        ctk.CTkLabel(
            self.main_frame, text="Mic A Input Level", font=ctk.CTkFont(size=12)
        ).pack(anchor="w", pady=(10, 0), padx=20)
        self.meter_in = ctk.CTkProgressBar(self.main_frame, width=450)
        self.meter_in.set(0)
        self.meter_in.pack(anchor="w", pady=(2, 5), padx=20)

        ctk.CTkLabel(
            self.main_frame,
            text="Processed Clean Output Level",
            font=ctk.CTkFont(size=12),
        ).pack(anchor="w", padx=20)
        self.meter_out = ctk.CTkProgressBar(
            self.main_frame, width=450, progress_color="#28a745"
        )
        self.meter_out.set(0)
        self.meter_out.pack(anchor="w", pady=(2, 5), padx=20)

    def get_devices(self, kind):
        devs = sd.query_devices()
        hostapis = sd.query_hostapis()
        mme_api_idx = next(
            (i for i, api in enumerate(hostapis) if api["name"] == "MME"), None
        )

        valid = {}
        for idx, d in enumerate(devs):
            if d[f"max_{kind}_channels"] > 0:
                if (
                    sys.platform == "win32"
                    and mme_api_idx is not None
                    and d["hostapi"] != mme_api_idx
                ):
                    continue
                name = f"{d['name']} (ID: {idx})"
                valid[name] = idx
        return valid

    def toggle_stream(self):
        if not self.streamer.running:
            a_name = self.combo_a.get()
            b_name = self.combo_b.get()
            out_name = self.out_combo.get()

            if not a_name or not b_name or not out_name:
                self.status_label.configure(
                    text="Error: Select all devices", text_color="#dc3545"
                )
                return

            id_a = self.in_devices[a_name]
            id_b = self.in_devices[b_name]
            id_out = self.out_devices[out_name]
            do_output = self.stream_output_var.get()

            self.status_label.configure(
                text="Initializing Dual-Mic DSP...", text_color="#ffc107"
            )
            self.update()

            success, msg = self.streamer.start_stream(id_a, id_b, id_out, do_output)
            if success:
                self.start_btn.configure(
                    text="Stop Streaming", fg_color="#dc3545", hover_color="#c82333"
                )
                self.status_label.configure(text=msg, text_color="#28a745")
            else:
                self.status_label.configure(text=f"Error: {msg}", text_color="#dc3545")
        else:
            self.streamer.stop_stream()
            self.start_btn.configure(
                text="Start Streaming", fg_color="#28a745", hover_color="#218838"
            )
            self.status_label.configure(text="Status: IDLE", text_color="gray")
            self.meter_in.set(0)
            self.meter_out.set(0)

    def update_meters(self, v_in, v_out):
        self.after(0, self._set_meters, v_in, v_out)

    def _set_meters(self, v_in, v_out):
        self.meter_in.set(v_in)
        self.meter_out.set(v_out)


if __name__ == "__main__":
    app = ModernLiveUI()
    app.mainloop()
