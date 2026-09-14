import os
import queue
import sys
import threading

import customtkinter as ctk
import numpy as np
import sounddevice as sd
import soxr

sys.path.append(os.path.abspath("D:/SIH/ANCMicModel/src"))
from highspl.dsp.nlms import RobustNLMS

# --- UI Theme Setup ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class AudioStreamer:
    def __init__(self):
        self.running = False
        self.stream_a = None
        self.stream_b = None
        self.stream_out = None

        self.queue_a = queue.Queue(maxsize=20)
        self.queue_b = queue.Queue(maxsize=20)

        self.nlms = RobustNLMS(
            pld_threshold_db=12.0,
            freeze_on_clipping=True,
            freeze_on_divergence=True,
            min_reference_energy=-40,
            max_reference_energy=-3,
        )
        self.state = self.nlms.reset()

        self.sr_a = 48000
        self.sr_b = 48000
        self.sr_out = 48000
        self.vol_callback = None
        self.shared_stereo = False

    def start_stream(self, mic_a_id, mic_b_id, output_id, enable_output):
        if self.running:
            return

        try:
            # Query devices
            info_a = sd.query_devices(mic_a_id, "input")
            info_b = sd.query_devices(mic_b_id, "input")
            info_out = sd.query_devices(output_id, "output")

            self.sr_a = int(info_a["default_samplerate"])
            self.sr_b = int(info_b["default_samplerate"])
            self.sr_out = int(info_out["default_samplerate"])

            self.state = self.nlms.reset()
            self.queue_a.queue.clear()
            self.queue_b.queue.clear()
            self.running = True

            self.shared_stereo = mic_a_id == mic_b_id

            self.process_thread = threading.Thread(
                target=self._processing_loop, args=(enable_output, output_id)
            )
            self.process_thread.start()

            if self.shared_stereo:
                # I2S Hardware / Single Stereo Interface (e.g. Raspberry Pi 5)
                self.stream_a = sd.InputStream(
                    device=mic_a_id,
                    channels=2,
                    samplerate=self.sr_a,
                    blocksize=512,
                    callback=self._stereo_callback,
                )
                self.stream_a.start()
            else:
                # Two separate USB Microphones (e.g. Windows Testing)
                self.stream_a = sd.InputStream(
                    device=mic_a_id,
                    channels=1,
                    samplerate=self.sr_a,
                    blocksize=512,
                    callback=self._mono_callback_a,
                )
                self.stream_b = sd.InputStream(
                    device=mic_b_id,
                    channels=1,
                    samplerate=self.sr_b,
                    blocksize=512,
                    callback=self._mono_callback_b,
                )
                self.stream_a.start()
                self.stream_b.start()

            return True, "Streaming Active"
        except Exception as e:  # noqa: BLE001
            self.running = False
            return False, str(e)

    def stop_stream(self):
        self.running = False
        if self.stream_a:
            self.stream_a.stop()
            self.stream_a.close()
        if self.stream_b:
            self.stream_b.stop()
            self.stream_b.close()
        if self.stream_out:
            self.stream_out.stop()
            self.stream_out.close()
        if hasattr(self, "process_thread"):
            self.process_thread.join(timeout=1.0)

    def _stereo_callback(self, indata, frames, time, status):
        if self.running:
            try:
                # Split Left (A) and Right (B) channels
                self.queue_a.put_nowait(indata[:, 0].copy())
                self.queue_b.put_nowait(
                    indata[:, 1].copy()
                    if indata.shape[1] > 1
                    else np.zeros_like(indata[:, 0])
                )
            except queue.Full:
                pass

    def _mono_callback_a(self, indata, frames, time, status):
        if self.running:
            try:
                self.queue_a.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass

    def _mono_callback_b(self, indata, frames, time, status):
        if self.running:
            try:
                self.queue_b.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass

    def _processing_loop(self, enable_output, output_id):
        if enable_output:
            self.stream_out = sd.OutputStream(
                device=output_id, channels=1, samplerate=self.sr_out, blocksize=0
            )
            self.stream_out.start()

        resamp_a = soxr.Resample(self.sr_a, 48000, 1) if self.sr_a != 48000 else None
        resamp_b = soxr.Resample(self.sr_b, 48000, 1) if self.sr_b != 48000 else None
        resamp_out = (
            soxr.Resample(48000, self.sr_out, 1) if self.sr_out != 48000 else None
        )

        while self.running:
            try:
                # Pull from both mics
                data_a = self.queue_a.get(timeout=0.1)
                data_b = self.queue_b.get(timeout=0.1)

                # Resample independently
                if resamp_a:
                    data_a = resamp_a.resample(data_a)
                if resamp_b:
                    data_b = resamp_b.resample(data_b)

                mic_a = data_a.astype(np.float32)
                mic_b = data_b.astype(np.float32)

                # DSP Processing
                out_48k, self.state = self.nlms.process(mic_a, mic_b, self.state)

                # UI Meters
                if self.vol_callback:
                    v_in = np.clip(np.max(np.abs(mic_a)), 0, 1)
                    v_out = np.clip(np.max(np.abs(out_48k)), 0, 1)
                    self.vol_callback(v_in, v_out)

                # Output
                if enable_output and self.stream_out:
                    out_48k = out_48k.reshape(-1, 1)
                    if resamp_out:
                        out_final = resamp_out.resample(out_48k)
                    else:
                        out_final = out_48k
                    self.stream_out.write(out_final)

            except queue.Empty:
                continue
            except Exception as e:  # noqa: BLE001
                print(f"Processing Error: {e}")
                break


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
            text="True Dual-Mic PLD + AI\nReal-Time Output Streaming",
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
