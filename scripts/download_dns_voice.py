import urllib.request
from pathlib import Path


def main():
    # DNS-Challenge hosts a few clean speech samples directly in their GitHub repo for testing.
    # We will download one of these so you don't have to download a 500 GB dataset!
    url = "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/tests/samples/ASR/spk1_snt1.wav"
    
    out_dir = Path("test_sample/urban_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / "human_voice.wav"
    
    print("Downloading clean human voice from Microsoft DNS-Challenge...")
    try:
        urllib.request.urlretrieve(url, str(out_path))
        print(f"✅ Successfully downloaded to: {out_path}")
        print("\nYour dual-mic test folder is now fully setup!")
        print(f"It contains:\n 1. {out_dir / 'garbage.wav'}\n 2. {out_dir / 'human_voice.wav'}")
        print("\nYou can now run the offline test:")
        print("uv run python scripts/run_pipeline.py")
    except Exception as e:  # noqa: BLE001
        print(f"❌ Failed to download: {e}")

if __name__ == "__main__":
    main()
