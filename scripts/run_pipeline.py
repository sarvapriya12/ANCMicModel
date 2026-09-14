import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from highspl.models.fastenhancer import FastEnhancerAdapter, FastEnhancerConfig


def main():
    parser = argparse.ArgumentParser(description="End-to-End Pipeline for Audio Enhancement")
    parser.add_argument("--test-dir", type=str, default="test_sample", help="Directory with input WAV files")
    parser.add_argument("--output-dir", type=str, default="output_sample", help="Directory for output WAV files")
    parser.add_argument("--backend", type=str, default="onnxruntime", choices=["pytorch", "onnxruntime"])
    
    args = parser.parse_args()
    
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    
    if not test_dir.exists():
        print(f"Error: Test directory '{test_dir}' does not exist.")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from highspl.dsp.nlms import RobustNLMS
    nlms_engine = RobustNLMS()
    
    print(f"Initializing model with backend: {args.backend}...")
    
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "models" / "fastenhancer" / "b" / "00500.pth"
    onnx_model = root / "models" / "fastenhancer" / "b" / "fastenhancer_b_streaming.onnx"
    
    adapter = FastEnhancerAdapter(
        FastEnhancerConfig(
            model_kwargs={
                "channels": 48,
                "kernel_size": [8, 3, 3],
                "stride": 4,
                "rnnformer_kwargs": {
                    "num_blocks": 3,
                    "channels": 36,
                    "freq": 36,
                    "num_heads": 4,
                    "eps": 1.0e-5,
                    "positional_embedding": "train",
                    "attn_bias": False,
                    "post_act": False,
                    "pre_norm": False,
                },
                "pre_post_init": "linear",
                "n_fft": 1024,
                "hop_size": 512,
                "win_size": 1024,
                "window": "hann",
                "stft_normalized": False,
                "mask": None,
                "activation": "SiLU",
                "activation_kwargs": {"inplace": True},
                "input_compression": 0.3,
                "normalize_final_conv": True,
                "weight_norm": True,
                "resnet": False,
                "sample_rate": 48_000,
            },
            checkpoint_path=str(checkpoint),
            onnx_path=str(onnx_model),
            backend=args.backend,
            device="cpu",
            upstream_root=str(root.parent / "fastenhancer"),
        )
    )
    
    print("Pipeline ready. Processing files...\n")
    
    def process_audio_file(audio_path, ref_path=None):
        try:
            audio, sr = sf.read(str(audio_path))
            if ref_path:
                ref_audio, ref_sr = sf.read(str(ref_path))
                if sr != ref_sr:
                    print("  Warning: Sample rate mismatch between primary and reference!")
        except Exception as e:  # noqa: BLE001
            print(f"  Failed to read audio: {e}")
            return
            
        if sr != 48000:
            import soxr
            print(f"  Resampling primary from {sr} Hz to 48000 Hz...")
            audio = soxr.resample(audio, sr, 48000)
            if ref_path:
                ref_audio = soxr.resample(ref_audio, ref_sr, 48000)
            sr = 48000
            
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        
        if ref_path:
            if ref_audio.ndim > 1:
                ref_audio = ref_audio.mean(axis=1)
            ref_audio = ref_audio.astype(np.float32)
            # Align lengths
            min_len = min(len(audio), len(ref_audio))
            audio = audio[:min_len]
            ref_audio = ref_audio[:min_len]
            
        fe_state = adapter.reset()
        nlms_state = nlms_engine.reset()
        outputs = []
        
        start_time = time.time()
        chunk_size = 512
        offset = 0
        
        while offset < audio.size:
            chunk_a = audio[offset : offset + chunk_size]
            if ref_path:
                chunk_b = ref_audio[offset : offset + chunk_size]
                if len(chunk_a) == len(chunk_b) and len(chunk_a) == chunk_size:
                    chunk_a, nlms_state = nlms_engine.process(chunk_a, chunk_b, nlms_state)
                    
            output, fe_state = adapter.process(chunk_a, sr, fe_state)
            if output.size:
                outputs.append(output)
            offset += chunk_size
            
        flushed, fe_state = adapter.flush(fe_state)
        if flushed.size:
            outputs.append(flushed)
            
        process_time = time.time() - start_time
        audio_duration = audio.size / sr
        rtf = process_time / audio_duration
        result = np.concatenate(outputs)
        
        return result, audio_duration, process_time, rtf

    # 1. Look for Dual-Mic subfolders
    subdirs = [d for d in test_dir.iterdir() if d.is_dir()]
    for d in subdirs:
        human_path = d / "human_voice.wav"
        garbage_path = d / "garbage.wav"
        
        if human_path.exists() and garbage_path.exists():
            print(f"Processing Dual-Mic Folder: {d.name} (NLMS + AI)")
            result, dur, comp, rtf = process_audio_file(human_path, garbage_path)
            
            out_folder = output_dir / d.name
            out_folder.mkdir(exist_ok=True)
            out_path = out_folder / "clean_output.wav"
            
            sf.write(str(out_path), result, 48000)
            print(f"  Saved to: {out_path.relative_to(output_dir)}")
            print(f"  Duration: {dur:.2f}s | Compute: {comp:.2f}s | RTF: {rtf:.3f}\n")
            
    # 2. Look for standalone single-mic files
    wav_files = []
    for ext in ("*.wav", "*.flac", "*.ogg"):
        wav_files.extend(test_dir.glob(ext))
        
    for wav_file in wav_files:
        if wav_file.parent == test_dir:  # Only root files, not inside subfolders
            print(f"Processing Single-Mic File: {wav_file.name} (AI Only)")
            result, dur, comp, rtf = process_audio_file(wav_file)
            
            out_path = output_dir / f"{wav_file.stem}_clean.wav"
            sf.write(str(out_path), result, 48000)
            print(f"  Saved to: {out_path.name}")
            print(f"  Duration: {dur:.2f}s | Compute: {comp:.2f}s | RTF: {rtf:.3f}\n")

if __name__ == "__main__":
    main()
