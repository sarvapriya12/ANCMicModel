import random
import shutil
from pathlib import Path

import soundata


def main():
    print("Initializing UrbanSound8K dataset via soundata...")
    # This initializes the dataset manager
    dataset = soundata.initialize('urbansound8k')
    
    print("\nDownloading dataset...")
    print("⚠️ WARNING: UrbanSound8K is approximately 6 GB in size.")
    print("Depending on your internet speed, this may take 10-30 minutes.")
    
    # Download the dataset (will save to ~/sound_datasets/urbansound8k by default)
    dataset.download()  
    
    print("\nValidating downloaded files...")
    dataset.validate()
    
    print("\nPicking a random heavy noise clip to use as 'garbage.wav'...")
    clip_ids = dataset.clip_ids
    
    # We want pure noise, so we will look for specific heavy noise classes
    noise_classes = ["air_conditioner", "car_horn", "drilling", "engine_idling", "jackhammer", "siren"]
    
    chosen_clip = None
    # Shuffle and find the first one that matches our desired noise classes
    for cid in random.sample(clip_ids, len(clip_ids)):
        clip = dataset.clip(cid)
        if clip.tags.labels[0] in noise_classes:
            chosen_clip = clip
            break
            
    if chosen_clip:
        class_name = chosen_clip.tags.labels[0]
        print(f"Selected Noise: {class_name} (ID: {chosen_clip.clip_id})")
        
        # Create the dual-mic test folder
        out_dir = Path("test_sample/urban_test")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        out_path = out_dir / "garbage.wav"
        shutil.copy(chosen_clip.audio_path, out_path)
        
        print(f"\n✅ Successfully copied to: {out_path}")
        print(f"You can now place a 'human_voice.wav' into '{out_dir}' and run the pipeline!")
    else:
        print("Could not find a suitable noise clip.")

if __name__ == "__main__":
    main()
