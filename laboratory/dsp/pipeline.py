import time
import numpy as np
import sys
import os

sys.path.append(os.path.abspath("D:/SIH/ANCMicModel/src"))
from highspl.dsp.nlms import RobustNLMS, VSSNLMS, NLMS
from highspl.dsp.fdaf import BattlefieldFDAF
from highspl.models.fastenhancer import FastEnhancerConfig, FastEnhancerAdapter

class HighSPLPipeline:
    def __init__(self):
        # Modes: "ROBUST", "VSS", "CLASSICAL", "FDAF", "BYPASS"
        self.dsp_mode = "ROBUST"
        
        # 1. Advanced Hard-Gated NLMS
        self.robust_nlms = RobustNLMS(pld_threshold_db=12.0, freeze_on_clipping=True, freeze_on_divergence=True)
        self.state_robust = self.robust_nlms.reset()
        
        # 2. Academic VSS-NLMS
        self.vss_nlms = VSSNLMS(filter_length=256, mu_max=0.5, mu_min=0.001, alpha=0.99, gamma=0.01)
        self.state_vss = self.vss_nlms.reset()
        
        # 3. Classical Baseline NLMS
        self.classical_nlms = NLMS(filter_length=256, step_size=0.25)
        self.state_classical = self.classical_nlms.reset()

        # 4. Battlefield FDAF
        self.battlefield_fdaf = BattlefieldFDAF(block_size=128, num_partitions=4, step_size=0.05)
        self.state_fdaf = self.battlefield_fdaf.reset()
        
        # Initialize FastEnhancer Neural Network
        fe_config = FastEnhancerConfig(
            model_kwargs={"hop_size": 512, "sample_rate": 48_000},
            onnx_path="D:/SIH/ANCMicModel/models/fastenhancer/b/fastenhancer_b_streaming.onnx",
            backend="onnxruntime",
            device="cpu"
        )
        self.fastenhancer = FastEnhancerAdapter(fe_config)
        self.state_fe = self.fastenhancer.reset()
        
    def process(self, primary: np.ndarray, reference: np.ndarray):
        eps = 1e-8
        p_energy = np.sum(primary**2)
        r_energy = np.sum(reference**2)
        cross = np.sum(primary * reference)
        coherence = abs(cross) / (np.sqrt(p_energy * r_energy) + eps)
        
        # --- STAGE 1: DSP (Linear Mathematics) ---
        t1 = time.perf_counter()
        
        if self.dsp_mode == "ROBUST":
            enhanced_audio_nlms, self.state_robust = self.robust_nlms.process(primary, reference, self.state_robust)
        elif self.dsp_mode == "VSS":
            enhanced_audio_nlms, self.state_vss = self.vss_nlms.process(primary, reference, self.state_vss)
        elif self.dsp_mode == "CLASSICAL":
            enhanced_audio_nlms, self.state_classical = self.classical_nlms.process(primary, reference, self.state_classical)
        elif self.dsp_mode == "FDAF":
            enhanced_audio_nlms, self.state_fdaf = self.battlefield_fdaf.process(primary, reference, self.state_fdaf)
        else:
            enhanced_audio_nlms = primary.copy()
            
        t2 = time.perf_counter()
        
        # --- STAGE 2: AI (Deep Learning Non-Linear Mask) ---
        enhanced_audio, self.state_fe = self.fastenhancer.process(
            enhanced_audio_nlms, sample_rate=48000, state=self.state_fe
        )
        
        # Apply 20% gain reduction as requested
        enhanced_audio = enhanced_audio * 0.8
        t3 = time.perf_counter()
        
        out_energy = np.sum(enhanced_audio**2)
        if out_energy < eps: out_energy = eps
        nr_db = 10 * np.log10(out_energy / (p_energy + eps))
        
        dsp_latency = (t2 - t1) * 1000
        ai_latency = (t3 - t2) * 1000
        
        return {
            "waveform": primary,
            "waveform_ref": reference,
            "enhanced_audio": enhanced_audio,
            "noise_reduction_db": nr_db,
            "coherence": coherence,
            "latency": {"dsp": dsp_latency, "ai": ai_latency}
        }
