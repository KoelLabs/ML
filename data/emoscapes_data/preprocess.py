import os
import zipfile
import librosa
import soundfile as sf
from pathlib import Path
import numpy as np
from collections import Counter


def extract_zip(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)


def normalize_audio(audio):
    rms = np.sqrt(np.mean(audio**2))
    return audio / (rms + 1e-8)


def process_audio_files(input_dir, output_dir, target_sr=16000):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    processed_counts = Counter()

    for category in ["human", "indicator", "mechanical", "nature", "quiet", "society"]:
        category_path = input_dir / "600_Sounds" / category
        if not category_path.exists():
            continue

        for audio_file in category_path.glob("*.wav"):
            try:
                audio, sr = librosa.load(audio_file, sr=target_sr)
                audio_normalized = normalize_audio(audio)
                output_filename = f"{category}_{audio_file.name}"
                output_path = output_dir / output_filename
                sf.write(output_path, audio_normalized, target_sr)
                processed_counts[category] += 1
                print(f"Processed: {output_filename}")
            except Exception as e:
                print(f"Error processing {audio_file.name}: {str(e)}")

    return processed_counts


if __name__ == "__main__":
    zip_path = "/home/arunasri/ML/notebooks/noise_samples/Emo-Soundscapes.zip"
    extract_path = "/home/arunasri/ML/notebooks/noise_samples/temp_extract"
    output_dir = "/home/arunasri/ML/notebooks/noise_samples/emo-soundscape"

    print("Extracting zip file...")
    extract_zip(zip_path, extract_path)

    print("\nProcessing files...")
    processed_counts = process_audio_files(
        Path(extract_path) / "Emo-Soundscapes" / "Emo-Soundscapes-Audio", output_dir
    )

    print("\nProcessed files per category:")
    for category, count in processed_counts.items():
        print(f"{category}: {count} files")

    with open(Path(output_dir) / "file_counts.txt", "w") as f:
        for category, count in processed_counts.items():
            f.write(f"{category}: {count}\n")

    # Cleanup temporary extraction directory
    import shutil

    shutil.rmtree(extract_path)
