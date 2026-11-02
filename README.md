# Meta SeamlessM4T v2 Speech-to-Text Benchmarking (Telugu Dialects - UROP)

This repository contains the dataset, scripts, model transcriptions, benchmark metrics, and evaluation reports for evaluating Meta's **SeamlessM4T v2 Large** (`facebook/seamless-m4t-v2-large`) foundation speech model on regional Telugu dialects.

---

## 📌 Project Overview
* **Dataset**: 900 speech recordings across **3 Telugu dialects** (*Rayalaseema*, *Srikakulam*, *Vizianagaram*) and **3 specialized domains** (*Dairy*, *Pharmacy*, *Textile*).
* **Model**: Meta SeamlessM4T v2 Large (2.3B parameters) via Hugging Face Transformers.
* **Hardware**: NVIDIA GeForce RTX 3060 (12 GB GDDR6), PyTorch FP16 with CUDA acceleration.
* **Corpus Results**:
  * **Word Error Rate (WER)**: 64.11%
  * **Character Error Rate (CER)**: 29.81%
  * **BLEU Score**: 18.48

---

## 📂 Repository Structure

```
├── urop/                                     # 900 Audio recordings (.ogg, .mp4, .mpeg, .mp3)
│   ├── rayalseema/                           # Rayalaseema dialect (Dairy, Pharmacy, Textile)
│   ├── srikakulam/                           # Srikakulam dialect (Dairy, Pharmacy, Textile)
│   └── vizianagaram/                         # Vizianagaram dialect (Dairy, Pharmacy, Textile)
├── ground_truths/                            # Reference sentence Excel files
│   ├── DairyGroundTruth_Reference_Sentences.xlsx
│   ├── PharmaGroundTruth_Reference_Sentences.xlsx
│   └── Textile_GroundTruth_Reference_Sentences.xlsx
├── transcriptions_seamlessm4t/               # 900 Individual text transcriptions (.txt)
├── transcribe_seamless.py                    # Audio decoding & GPU inference pipeline
├── seamlessm4t_transcriptions.csv            # Consolidated table of all 900 transcriptions
├── evaluation_metrics_detailed.csv           # Sentence-level WER, CER, and BLEU evaluation
├── METHODOLOGY_AND_EXPERIMENT_STEPS.md       # Step-by-step workflow & execution report
├── SEAMLESSM4T_EVALUATION_REPORT.md          # Full analytical evaluation report
├── SEAMLESSM4T_EVALUATION_REPORT.html        # Styled HTML report
└── SEAMLESSM4T_EVALUATION_REPORT.pdf         # Printable publication-quality PDF report
```

---

## 🚀 Step-by-Step Methodology
For the detailed chronological report documenting every phase from model downloading and resilient audio decoding to GPU inference and metric calculation, see:
👉 [**METHODOLOGY_AND_EXPERIMENT_STEPS.md**](METHODOLOGY_AND_EXPERIMENT_STEPS.md)

For the detailed linguistic and dialect evaluation report with error breakdown:
👉 [**SEAMLESSM4T_EVALUATION_REPORT.md**](SEAMLESSM4T_EVALUATION_REPORT.md)

---

## 🛠️ How to Run Inference

### 1. Requirements
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers soundfile librosa av pandas numpy tqdm openpyxl sacrebleu jiwer
```

### 2. Run Transcription
```bash
python transcribe_seamless.py \
    --audio_dir ./urop \
    --output_csv ./seamlessm4t_transcriptions.csv \
    --output_dir ./transcriptions_seamlessm4t \
    --tgt_lang tel \
    --model_id facebook/seamless-m4t-v2-large \
    --device cuda
```

---

## 📊 Key Results Breakdown

| Dialect Region | Sample Count | WER (%) ↓ | CER (%) ↓ | BLEU Score ↑ |
| :--- | :---: | :---: | :---: | :---: |
| **Rayalaseema** | 301 | **61.02%** | **25.44%** | **19.39** |
| **Vizianagaram** | 300 | **64.21%** | **30.71%** | **19.73** |
| **Srikakulam** | 299 | **67.10%** | **33.29%** | **16.23** |

| Domain / Category | Sample Count | WER (%) ↓ | CER (%) ↓ | BLEU Score ↑ |
| :--- | :---: | :---: | :---: | :---: |
| **Dairy** | 299 | **55.36%** | **21.45%** | **24.67** |
| **Textile** | 300 | **61.84%** | **27.56%** | **19.96** |
| **Pharmacy** | 301 | **74.48%** | **39.70%** | **11.13** |
