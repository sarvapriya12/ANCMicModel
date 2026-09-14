import sounddevice as sd
import numpy as np
import collections
import time
import json
import soundfile as sf
import os

class AudioCapture:
    def __init__(self, mic_a_id, mic_b_id, output_id, blocksize=512, sr=48000):
        self.mic_a_id = mic_a_id
        self.mic_b_id = mic_b_id
        self.output_id = output_id
        self.blocksize = blocksize
        self.sr = sr
        
        self.running = False
        self.buffer_size = 20
        self.queue_a = collections.deque(maxlen=self.buffer_size)
        self.queue_b = collections.deque(maxlen=self.buffer_size)
        self.out_queue = collections.deque(maxlen=self.buffer_size)
        
        self.stream_a = None
        self.stream_b = None
        self.stream_out = None
        self.shared_stereo = (mic_a_id == mic_b_id)
        
        self.recording = False
        self.record_path = "D:/SIH/ANCMicModel/laboratory"
        self.rec_a = []
        self.rec_b = []
        self.rec_enh = []
        self.rec_metrics = []

    def _stereo_cb(self, indata, frames, time_info, status):
        if status: print(f"[Audio Error] {status}")
        if self.running:
            self.queue_a.append(indata[:, 0].copy())
            self.queue_b.append(indata[:, 1].copy() if indata.shape[1] > 1 else np.zeros_like(indata[:, 0]))

    def _mono_cb_a(self, indata, frames, time_info, status):
        if status: print(f"[Audio Error A] {status}")
        if self.running:
            self.queue_a.append(indata[:, 0].copy())

    def _mono_cb_b(self, indata, frames, time_info, status):
        if status: print(f"[Audio Error B] {status}")
        if self.running:
            self.queue_b.append(indata[:, 0].copy())

    def _out_cb(self, outdata, frames, time_info, status):
        if status: print(f"[Audio Error Out] {status}")
        if self.running:
            try:
                data = self.out_queue.popleft()
                if len(data) == frames:
                    outdata[:, 0] = data
                else:
                    outdata.fill(0)
            except IndexError:
                outdata.fill(0)

    @staticmethod
    def _validate_device(dev_id, direction="input"):
        """Check that a device ID is valid and has the required channels."""
        try:
            devs = sd.query_devices()
            if dev_id is None:
                return False, "No device selected (ID is None)"
            if dev_id < 0 or dev_id >= len(devs):
                return False, f"Device ID {dev_id} is out of range (system has {len(devs)} devices)"
            dev = devs[dev_id]
            key = "max_input_channels" if direction == "input" else "max_output_channels"
            if dev[key] < 1:
                return False, f"Device [{dev_id}] '{dev['name']}' has no {direction} channels"
            return True, f"Device [{dev_id}] '{dev['name']}' OK"
        except Exception as e:
            return False, f"Device query failed: {e}"

    def start(self):
        self.running = True
        self.queue_a.clear()
        self.queue_b.clear()
        self.out_queue.clear()

        # --- Validate all devices before opening any streams ---
        ok_a, msg_a = self._validate_device(self.mic_a_id, "input")
        if not ok_a:
            print(f"[AUDIO ERROR] Mic A: {msg_a}")
            self.running = False
            raise RuntimeError(f"Mic A device invalid: {msg_a}")

        ok_b, msg_b = self._validate_device(self.mic_b_id, "input")
        if not ok_b:
            print(f"[AUDIO ERROR] Mic B: {msg_b}")
            self.running = False
            raise RuntimeError(f"Mic B device invalid: {msg_b}")

        ok_out, msg_out = self._validate_device(self.output_id, "output")
        if not ok_out:
            print(f"[AUDIO ERROR] Output: {msg_out}")
            self.running = False
            raise RuntimeError(f"Output device invalid: {msg_out}")

        print(f"[AUDIO] Mic A: {msg_a}")
        print(f"[AUDIO] Mic B: {msg_b}")
        print(f"[AUDIO] Output: {msg_out}")

        try:
            if self.shared_stereo:
                dev_info = sd.query_devices(self.mic_a_id)
                channels = min(2, dev_info["max_input_channels"])
                self.stream_a = sd.InputStream(
                    device=self.mic_a_id, channels=channels,
                    samplerate=self.sr, blocksize=self.blocksize,
                    callback=self._stereo_cb if channels >= 2 else self._mono_cb_a,
                )
                self.stream_a.start()
                if channels < 2:
                    print("[AUDIO WARNING] Shared stereo requested but device only has 1 channel. "
                          "Mic B will receive zeros (mono fallback).")
            else:
                self.stream_a = sd.InputStream(
                    device=self.mic_a_id, channels=1,
                    samplerate=self.sr, blocksize=self.blocksize,
                    callback=self._mono_cb_a,
                )
                self.stream_b = sd.InputStream(
                    device=self.mic_b_id, channels=1,
                    samplerate=self.sr, blocksize=self.blocksize,
                    callback=self._mono_cb_b,
                )
                self.stream_a.start()
                self.stream_b.start()

            self.stream_out = sd.OutputStream(
                device=self.output_id, channels=1,
                samplerate=self.sr, blocksize=self.blocksize,
                callback=self._out_cb,
            )
            self.stream_out.start()
            print(f"[AUDIO] All streams started successfully (sr={self.sr}, bs={self.blocksize})")
        except Exception as e:
            print(f"[AUDIO ERROR] Failed to start streams: {e}")
            self.stop()
            raise

    def stop(self):
        self.running = False
        if self.stream_a: self.stream_a.stop(); self.stream_a.close()
        if self.stream_b: self.stream_b.stop(); self.stream_b.close()
        if self.stream_out: self.stream_out.stop(); self.stream_out.close()
        if self.recording:
            self.save_session()

    def get_frames(self):
        retries = 10
        while retries > 0:
            try:
                a = self.queue_a.popleft()
                b = self.queue_b.popleft()
                return a, b
            except IndexError:
                time.sleep(0.01)
                retries -= 1
        return None, None
            
    def push_output(self, enhanced_audio, metrics=None):
        self.out_queue.append(enhanced_audio)
        if self.recording and metrics:
            self.rec_enh.append(enhanced_audio.copy())
            self.rec_a.append(metrics.get("waveform", np.zeros_like(enhanced_audio)).copy())
            self.rec_b.append(metrics.get("waveform_ref", np.zeros_like(enhanced_audio)).copy())
            self.rec_metrics.append({
                "noise_reduction_db": float(metrics.get("noise_reduction_db", 0)),
                "coherence": float(metrics.get("coherence", 0)),
                "latency": metrics.get("latency", {})
            })

    def start_recording(self):
        self.recording = True
        self.rec_a = []
        self.rec_b = []
        self.rec_enh = []
        self.rec_metrics = []
        
    def save_session(self):
        self.recording = False
        if not self.rec_enh: return
        
        try:
            a_data = np.concatenate(self.rec_a)
            b_data = np.concatenate(self.rec_b)
            enh_data = np.concatenate(self.rec_enh)
            
            sf.write(os.path.join(self.record_path, "mic_a.wav"), a_data, self.sr)
            sf.write(os.path.join(self.record_path, "mic_b.wav"), b_data, self.sr)
            sf.write(os.path.join(self.record_path, "enhanced.wav"), enh_data, self.sr)
            
            with open(os.path.join(self.record_path, "metrics.json"), "w") as f:
                json.dump(self.rec_metrics, f, indent=2)
                
            print(f"[SAVE] Session saved to {self.record_path} (mic_a.wav, mic_b.wav, enhanced.wav, metrics.json)")
        except Exception as e:
            print(f"[SAVE ERROR] {e}")
