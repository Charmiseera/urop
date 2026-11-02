import os
import sys

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure HF cache is stored on D: drive to protect C: drive space
os.environ["HF_HOME"] = r"D:\huggingface_cache"

import glob
import argparse
import time
import pandas as pd
import numpy as np
import torch
import soundfile as sf
import librosa
import av
from tqdm import tqdm
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

def load_audio(file_path, target_sr=16000):
    """Robust audio loader supporting ogg, mpeg, mp4, wav, flac, and noisy trailing-byte streams."""
    # 1. Try PyAV safe packet-level decoding first (most resilient against stream truncations)
    try:
        container = av.open(file_path)
        audio_stream = next(s for s in container.streams if s.type == 'audio')
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=target_sr)
        frames = []
        try:
            for packet in container.demux(audio_stream):
                try:
                    for frame in packet.decode():
                        for resampled_frame in resampler.resample(frame):
                            frames.append(resampled_frame.to_ndarray())
                except Exception:
                    continue
        finally:
            container.close()
        if frames:
            return np.concatenate(frames, axis=1).squeeze(0).astype(np.float32)
    except Exception:
        pass

    # 2. Try soundfile
    try:
        data, sr = sf.read(file_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        return data.astype(np.float32)
    except Exception:
        pass

    # 3. Try librosa
    try:
        data, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return data.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to decode audio file {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using SeamlessM4T v2")
    parser.add_argument("--audio_dir", type=str, default=r"D:\New folder\urop", help="Root directory containing audio files")
    parser.add_argument("--output_csv", type=str, default=r"D:\New folder\seamlessm4t_transcriptions.csv", help="Path to output CSV")
    parser.add_argument("--output_dir", type=str, default=r"D:\New folder\transcriptions_seamlessm4t", help="Directory for individual text files")
    parser.add_argument("--tgt_lang", type=str, default="tel", help="Target transcription language ('tel' for Telugu, 'eng' for English)")
    parser.add_argument("--model_id", type=str, default="facebook/seamless-m4t-v2-large", help="Hugging Face model ID")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    args = parser.parse_args()

    print(f"=== SeamlessM4T Speech-to-Text Transcription ===")
    print(f"Audio Directory : {args.audio_dir}")
    print(f"Output CSV      : {args.output_csv}")
    print(f"Output Text Dir : {args.output_dir}")
    print(f"Target Language : {args.tgt_lang} (Telugu)")
    print(f"Device          : {args.device}")
    print(f"Model ID        : {args.model_id}")

    # Gather audio files (all common extensions)
    extensions = ("*.ogg", "*.wav", "*.mp3", "*.mpeg", "*.mp4", "*.m4a", "*.flac")
    audio_files = []
    for ext in extensions:
        audio_files.extend(glob.glob(os.path.join(args.audio_dir, "**", ext), recursive=True))
    
    audio_files = sorted(list(set(audio_files)))
    print(f"Found {len(audio_files)} audio files.")

    if not audio_files:
        print("No audio files found yet. Waiting for files to finish downloading...")
        return

    # Check for existing transcriptions to support resume
    existing_records = {}
    if os.path.exists(args.output_csv):
        try:
            df_prev = pd.read_csv(args.output_csv, encoding="utf-8-sig")
            for _, row in df_prev.iterrows():
                # Only cache if successful and non-empty
                if row.get("status") == "success" and pd.notna(row.get("transcription")) and str(row.get("transcription")).strip():
                    existing_records[row["file_path"]] = str(row["transcription"]).strip()
            print(f"Found {len(existing_records)} valid existing transcriptions in {args.output_csv} (will skip).")
        except Exception as e:
            print(f"Notice: Could not load previous CSV: {e}")

    # Load processor and model
    print(f"\nLoading processor and model from {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    
    dtype = torch.float16 if "cuda" in args.device else torch.float32
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        args.model_id, 
        torch_dtype=dtype
    ).to(args.device)
    model.eval()
    print("Model ready on GPU.\n")

    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    # Process files
    pbar = tqdm(audio_files, desc="Transcribing")
    for file_path in pbar:
        rel_path = os.path.relpath(file_path, args.audio_dir)
        path_parts = rel_path.split(os.sep)
        
        # Extract dialect and category from directory hierarchy
        dialect = path_parts[0] if len(path_parts) > 1 else "unknown"
        category = path_parts[1] if len(path_parts) > 2 else "unknown"
        filename = os.path.basename(file_path)

        # Check if already processed
        if file_path in existing_records:
            transcription = existing_records[file_path]
            results.append({
                "dialect": dialect,
                "category": category,
                "filename": filename,
                "file_path": file_path,
                "rel_path": rel_path,
                "transcription": transcription,
                "status": "success"
            })
            continue

        try:
            # Load audio (16kHz mono)
            audio_array = load_audio(file_path, target_sr=16000)
            audio_inputs = processor(audio=audio_array, sampling_rate=16000, return_tensors="pt")
            audio_inputs = {k: v.to(args.device) for k, v in audio_inputs.items()}

            with torch.no_grad():
                output_tokens = model.generate(**audio_inputs, tgt_lang=args.tgt_lang)[0].cpu().squeeze()
            
            transcription = processor.decode(output_tokens.tolist(), skip_special_tokens=True).strip()

            # Save individual .txt file
            text_out_path = os.path.join(args.output_dir, rel_path + ".txt")
            os.makedirs(os.path.dirname(text_out_path), exist_ok=True)
            with open(text_out_path, "w", encoding="utf-8") as f:
                f.write(transcription)

            results.append({
                "dialect": dialect,
                "category": category,
                "filename": filename,
                "file_path": file_path,
                "rel_path": rel_path,
                "transcription": transcription,
                "status": "success"
            })
            pbar.set_postfix_str(f"{dialect}/{category} | {transcription[:15]}...")

        except Exception as e:
            print(f"\nError processing {file_path}: {e}")
            results.append({
                "dialect": dialect,
                "category": category,
                "filename": filename,
                "file_path": file_path,
                "rel_path": rel_path,
                "transcription": "",
                "status": f"error: {str(e)}"
            })

        # Save progress incrementally to CSV every 10 files
        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    # Final save
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"\nAll done! Successfully saved {len(results)} transcriptions to:")
    print(f" -> CSV  : {args.output_csv}")
    print(f" -> Text : {args.output_dir}")

if __name__ == "__main__":
    main()
