# Meta SeamlessM4T v2: Speech-to-Text Evaluation & Comparison Report
**Telugu Regional Dialect Speech Dataset (UROP)**

---

## 1. Executive Summary

This report evaluates Meta's **SeamlessM4T v2 Large** (`facebook/seamless-m4t-v2-large`) on a multi-dialect Telugu speech recognition dataset comprising **900 audio recordings** across **3 dialect regions** (*Rayalaseema*, *Srikakulam*, *Vizianagaram*) and **3 specialized domains** (*Dairy*, *Pharmacy*, *Textile*).

The evaluation benchmarks the model's generated Telugu transcriptions against standardized ground-truth reference texts using three standard evaluation metrics:
* **Word Error Rate (WER)**: Word-level edit distance.
* **Character Error Rate (CER)**: Character/akshara-level phonetic edit distance.
* **BLEU (SacreBLEU)**: Exact n-gram overlap precision.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CORPUS-WIDE RESULTS                             │
│                                                                        │
│     Word Error Rate (WER)    :  64.11%                                 │
│     Character Error Rate (CER):  29.81%                                 │
│     BLEU Score               :  18.48                                  │
│     Total Audio Evaluated    :  900 files (100% evaluated)             │
│     Average Duration / Audio :  ~3.8 - 5.5 seconds                     │
└────────────────────────────────────────────────────────────────────────┘
```

> [!NOTE]
> The **Character Error Rate (29.81%)** is substantially lower than the **Word Error Rate (64.11%)**, indicating that SeamlessM4T v2 captures the phonetic sequence of spoken Telugu with high fidelity. The divergence between WER and CER stems primarily from Telugu word agglutination (*Sandhi*), variations in English technical loanword transliteration, and reference text formatting artifacts.

---

## 2. Experimental Setup & System Architecture

### 2.1 Hardware & Runtime Environment
* **GPU**: NVIDIA GeForce RTX 3060 (12 GB GDDR6 VRAM)
* **Compute Architecture**: PyTorch `2.11.0+cu128` (CUDA Acceleration enabled)
* **Precision**: Half-precision (`torch.float16`)
* **Audio Sampling Rate**: Resampled to 16,000 Hz Mono using a hybrid decoding engine (`soundfile` + `PyAV` + `librosa`)
* **Storage Allocation**: Weights and caching isolated to `D:\huggingface_cache`

### 2.2 Dataset Distribution
The dataset contains 900 files divided into regional dialects and domains:

| Dialect Region | Dairy | Pharmacy | Textile | Regional Total |
| :--- | :---: | :---: | :---: | :---: |
| **Rayalaseema** | 100 | 101* | 100 | **301** |
| **Srikakulam** | 99** | 100 | 100 | **299** |
| **Vizianagaram** | 100 | 100 | 100 | **300** |
| **Domain Total** | **299** | **301** | **300** | **900** |

*\* Prompt #95 in Rayalaseema Pharmacy was recorded in two formats (`95.pharmacy.mpeg` and `95.pharmacy.ogg`). Both were independently evaluated.*  
*\*\* Prompt #95 in Srikakulam Dairy was omitted from the source upload.*

---

## 3. Benchmark Metrics & Performance Breakdown

### 3.1 Dialect-Wise Performance
Different regional speech habits and phonological features significantly impacted recognition accuracy:

| Dialect Region | Sample Count | WER (%) ↓ | CER (%) ↓ | BLEU Score ↑ |
| :--- | :---: | :---: | :---: | :---: |
| **Rayalaseema** | 301 | **61.02%** | **25.44%** | **19.39** |
| **Vizianagaram** | 300 | **64.21%** | **30.71%** | **19.73** |
| **Srikakulam** | 299 | **67.10%** | **33.29%** | **16.23** |

```mermaid
pie title Error Distribution by Dialect Region (Relative CER)
    "Rayalaseema (25.4%)" : 25.44
    "Vizianagaram (30.7%)" : 30.71
    "Srikakulam (33.3%)" : 33.29
```

* **Rayalaseema**: Yielded the lowest error rates (CER 25.44%, WER 61.02%). The dialect exhibits phonetic clarity and standard grammatical structure aligning well with SeamlessM4T's training priors.
* **Vizianagaram**: Moderate performance with competitive BLEU (19.73).
* **Srikakulam**: Exhibited the highest error rates across all metrics (WER 67.10%, CER 33.29%). This is attributable to northern coastal Andhra phonetic shifts, distinctive cadence, and localized colloquialisms.

---

### 3.2 Domain-Wise Performance
The acoustic and vocabulary differences between everyday agricultural terms and specialized medical/technical terms created notable variance across domains:

| Domain / Category | Sample Count | WER (%) ↓ | CER (%) ↓ | BLEU Score ↑ |
| :--- | :---: | :---: | :---: | :---: |
| **Dairy** | 299 | **55.36%** | **21.45%** | **24.67** |
| **Textile** | 300 | **61.84%** | **27.56%** | **19.96** |
| **Pharmacy** | 301 | **74.48%** | **39.70%** | **11.13** |

* **Dairy achieved the highest accuracy** (WER 55.36%, CER 21.45%, BLEU 24.67) due to straightforward agrarian vocabulary (*ఆవు*, *మిల్క్*, *దూడ*, *వెటర్నరీ*).
* **Pharmacy had the highest error rate** (WER 74.48%, CER 39.70%), caused by technical English pharmaceutical loanwords (*inhaler*, *narcotic register*, *refrigeration*, *generic medicine*) which vary in pronunciation and transliteration.

---

### 3.3 Full 3×3 Performance Matrix (Dialect × Domain)

| Dialect | Domain | File Count | WER (%) ↓ | CER (%) ↓ | BLEU Score ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Rayalaseema** | **Dairy** | 100 | **49.65%** | **17.44%** | **27.94** |
| **Rayalaseema** | **Pharmacy** | 101 | **70.47%** | **32.57%** | **11.71** |
| **Rayalaseema** | **Textile** | 100 | **62.09%** | **25.65%** | **18.96** |
| **Srikakulam** | **Dairy** | 99 | **62.13%** | **27.44%** | **20.58** |
| **Srikakulam** | **Pharmacy** | 100 | **72.80%** | **37.68%** | **11.24** |
| **Srikakulam** | **Textile** | 100 | **66.01%** | **34.30%** | **17.08** |
| **Vizianagaram** | **Dairy** | 100 | **54.37%** | **19.52%** | **25.41** |
| **Vizianagaram** | **Pharmacy** | 100 | **80.21%** | **48.92%** | **10.39** |
| **Vizianagaram** | **Textile** | 100 | **57.42%** | **22.73%** | **23.72** |

---

## 4. In-Depth Linguistic & Error Analysis

Analyzing the paired predictions against reference sentences in [`evaluation_metrics_detailed.csv`](file:///D:/New%20folder/evaluation_metrics_detailed.csv) reveals four distinct sources of error:

### 4.1 Telugu Compound Words & Agglutination (*Sandhi*)
Telugu is an agglutinative language where postpositions and auxiliary verbs blend with preceding words. While the speaker produces a continuous sound, human transcribers often separate tokens arbitrarily:

| Audio ID | Ground Truth Reference | SeamlessM4T Hypothesis | Error Cause |
| :--- | :--- | :--- | :--- |
| `1.dairy` | ... బాగుంది **అని** ల్యాబ్ రిపోర్ట్. | ... **బాగుందని** లాబ్రిగోట్ చెప్పింది | Model combined *బాగుంది + అని* into *బాగుందని*. Phonetically and grammatically valid, but penalized by WER. |
| `14.dairy` | ... కొత మెషిన్ తెచాము. | ... కొత్త మెషిన్ **తీసుకొని వచ్చాము** | Verbal periphrasis (*తీసుకొచ్చాము* vs *తెచ్చాము*). |
| `29.dairy` | ... **ప్రతి రోజు** మైంటైన్ చేస్తాము. | ... **ప్రతిరోజూ** మెయింటైన్ చేస్తాము | Space insertion vs continuous spelling. |

### 4.2 Reference Text Typographical Errors (Model Correction)
In multiple cases, **SeamlessM4T transcribed the audio more accurately than the ground truth in the Excel file**:

| Audio ID | Ground Truth (Contains Typo) | SeamlessM4T (Correct Telugu) | Analysis |
| :--- | :--- | :--- | :--- |
| `10.dairy` | ... సరిగా **జరింది**. | ... సరిగ్గా **జరిగింది**. | Ground truth omitted *గ*. Model produced correct orthography. |
| `11.dairy` | ... **కొత** మెషిన్ **తెచాము**. | ... **కొత్త** మిషన్ **తెచ్చాము**. | Ground truth missed double consonants (*త్*, *చ్*). |
| `13.dairy` | ... **పకారం** ప్రొడక్షన్ **పెరింది**. | ... **ప్రకారం** ప్రొడక్షన్ **పెరిగింది**. | Ground truth had missing consonant clusters (*ప్ర*, *గి*). |
| `24.dairy` | ... రిపోర్ట్ **రిపోర్ట్** పకారం ... | ... రిపోర్ట్ ప్రకారం ... | Ground truth had duplicate word repetition. |

### 4.3 Ground Truth Syllable-Spacing Artifacts (Pharma & Textile)
In the uploaded Pharmacy and Textile Excel sheets, many words were tokenized with internal spaces before matras and anusvara characters:
* Reference: `ఈ  ఇన్హేలర్  యొక్క రిఫ్రిజిరేషన్  చెక్  చేయమని ఫార్మసిస్ట్  చెప్పా రు.` (`చెప్పా` + `రు`)
* Reference: `స్టోర్  మేనేజర్ ... గురిం చి రివ్యూ చేసారు.` (`గురిం` + `చి`)
* Reference: `... కి సం బం ధిం చిన ... అప్డేట్ అయ్యిం య్యిం ది.` (`సం బం ధిం చిన`)
* SeamlessM4T output standard, unbroken words (`చెప్పారు`, `గురించి`, `సంబంధించిన`), which caused whitespace-based tokenizers to flag token count mismatches.

### 4.4 English Language Drift on Ambiguous Audio
In file `22.dairy.ogg` (*మా ఆవు యొక సైలేజ్ ఈ రోజు రికార్డ్ చేసాము.*), the speaker's accent on English technical words (*"silage"*, *"record"*) caused the model's unconstrained decoder to switch to English output:
* **Hypothesis**: `"Today, I will record the silhouette of Mao."`
* **Sentence WER**: 100.0% | **Sentence BLEU**: 0.0

---

## 5. Performance Optimization Roadmap

To reduce the WER from **64% down to 25–35%**, the following optimizations are recommended:

```mermaid
graph TD
    A[Current Baseline: 64.11% WER] --> B[Phase 1: Preprocessing & Decoding]
    B --> C[Expected WER: ~52-55%]
    C --> D[Phase 2: Text Normalization]
    D --> E[Expected WER: ~40-45%]
    E --> F[Phase 3: LoRA Fine-Tuning on RTX 3060]
    F --> G[Target WER: 25-32%]
```

### Phase 1: Audio Preprocessing & Constrained Decoding (Immediate)
1. **Voice Activity Detection (VAD) & Silence Trimming**: Trimming leading/trailing dead air using `librosa.effects.trim(top_db=25)` prevents hallucination and language-switching loops on low-volume recordings.
2. **Beam Search Tuning**: Replace greedy decoding with `num_beams=4` and `length_penalty=1.0` to stabilize decoding on dialectal accents.
3. **Language Consistency Validation**: If an output contains non-Telugu script, automatically fall back to temperature-scaled greedy decoding with forced Telugu prefix tokens.

### Phase 2: Text & Sandhi Normalization
1. **Telugu Sandhi Normalizer**: Implement an Indic text normalizer to map common split/joint suffixes (*బాగుంది అని* $\leftrightarrow$ *బాగుందని*, *ప్రతి రోజు* $\leftrightarrow$ *ప్రతిరోజూ*).
2. **Ground Truth Cleaning**: Clean the syllable spaces in Pharmacy and Textile references (*గురిం చి* $\rightarrow$ *గురించి*, *చెప్పా రు* $\rightarrow$ *చెప్పారు*).

### Phase 3: LoRA (Low-Rank Adaptation) Fine-Tuning
* Train a lightweight adapter (`r=16, lora_alpha=32`) targeting the attention projection layers (`q_proj`, `v_proj`) of SeamlessM4T v2.
* **Hardware Suitability**: With 12 GB VRAM on the RTX 3060 and 4-bit / 8-bit quantized weights (`bitsandbytes`), fine-tuning the 900 samples takes **~20 to 30 minutes**.
* **Impact**: Adapts the acoustic encoder directly to Rayalaseema, Srikakulam, and Vizianagaram acoustic patterns, significantly reducing out-of-vocabulary and loanword transliteration errors.

---

## 6. Generated Project Deliverables

| Deliverable | File Path | Description |
| :--- | :--- | :--- |
| **Evaluation CSV** | [`D:\New folder\evaluation_metrics_detailed.csv`](file:///D:/New%20folder/evaluation_metrics_detailed.csv) | Full 900-row file with paired Ground Truth, Hypothesis, sentence-level WER, CER, and BLEU. |
| **Model Transcriptions** | [`D:\New folder\seamlessm4t_transcriptions.csv`](file:///D:/New%20folder/seamlessm4t_transcriptions.csv) | Raw hypothesis transcriptions with metadata. |
| **Text Output Directory** | [`D:\New folder\transcriptions_seamlessm4t`](file:///D:/New%20folder/transcriptions_seamlessm4t) | Individual `.txt` files matching the audio directory hierarchy. |
| **Inference Script** | [`D:\New folder\transcribe_seamless.py`](file:///D:/New%20folder/transcribe_seamless.py) | GPU-accelerated pipeline with resilient multi-format audio decoding. |
