import os
import time
from pathlib import Path

import numpy as np
import psutil
import sounddevice as sd
import soundfile as sf

from highspl.models.fastenhancer import FastEnhancerAdapter, FastEnhancerConfig


def main():
    duration = 40  # seconds
    sr = 48000

    print("=" * 50)
    print("🎙️  LIVE MICROPHONE TEST 🎙️")
    print("=" * 50)

    print(f"\nRecording {duration} seconds from your default microphone...")
    print("Speak now! (Try making some background noise too)")

    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")

    for i in range(duration, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    sd.wait()
    print("\n✅ Recording finished!")

    # Save the raw noisy file
    test_dir = Path("test_sample")
    test_dir.mkdir(exist_ok=True)
    raw_path = test_dir / "live_noisy.wav"
    sf.write(str(raw_path), audio, sr)
    print(f"Saved raw audio to: {raw_path}")

    # Process it
    print("\n⚙️  Processing through FastEnhancer ONNX...")
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "models" / "fastenhancer" / "b" / "00500.pth"
    onnx_model = (
        root / "models" / "fastenhancer" / "b" / "fastenhancer_b_streaming.onnx"
    )

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
            backend="onnxruntime",
            device="cpu",
            upstream_root=str(root.parent / "fastenhancer"),
        )
    )

    flat_audio = audio.reshape(-1)

    # Normalize the audio to peak at 0.9 before passing to the model
    # This prevents extreme over-suppression on quiet microphones!
    max_val = np.max(np.abs(flat_audio))
    if max_val > 0:
        flat_audio = (flat_audio / max_val) * 0.9

    state = adapter.reset()
    outputs = []

    chunk_size = 512
    offset = 0
    start_time = time.time()

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    peak_mem = mem_before

    while offset < flat_audio.size:
        chunk = flat_audio[offset : offset + chunk_size]
        output, state = adapter.process(chunk, sr, state)
        if output.size:
            outputs.append(output)
        offset += chunk_size

        current_mem = process.memory_info().rss / (1024 * 1024)
        peak_mem = max(peak_mem, current_mem)

    flushed, state = adapter.flush(state)
    if flushed.size:
        outputs.append(flushed)

    compute_time = time.time() - start_time
    clean_audio = np.concatenate(outputs)
    rtf = compute_time / duration
    print("✅ Processing complete!")
    print(f"   Compute time : {compute_time:.2f}s")
    print(f"   RTF          : {rtf:.3f}")
    print(f"   RAM Usage    : Peak {peak_mem:.2f} MB (Started at {mem_before:.2f} MB)")

    out_dir = Path("output_sample")
    out_dir.mkdir(exist_ok=True)
    clean_path = out_dir / "live_clean.wav"
    sf.write(str(clean_path), clean_audio, sr)
    print(f"Saved cleaned audio to: {clean_path}")

    print("\n🔊 Playing back the ORIGINAL NOISY recording...")
    sd.play(audio, sr)
    sd.wait()

    print("\n✨ Playing back the CLEANED recording...")
    sd.play(clean_audio, sr)
    sd.wait()

    print("\nDone! Check test_sample/ and output_sample/ to listen again.")


if __name__ == "__main__":
    main()
