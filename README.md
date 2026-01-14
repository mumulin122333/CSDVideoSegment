# CSDVideoSegment

CSDVideoSegment is an end-to-end pipeline for **case-based endoscopy video segmentation**.

The pipeline automatically performs:

- Video → frame extraction
- Frame-level filtering and voting using fine-tuned timm models
- Phase inference using Vision-Language Models (VLM) via vLLM
- Final phase-level CSV report generation

Users only need to prepare **videos and case IDs**.
All intermediate files are generated automatically and can be safely ignored.

---

## 1. Environment Setup

### 1.1 Create Conda Environment

```bash
conda create -n CSDVideoSegment python=3.11 -y
conda activate CSDVideoSegment
```

---

### 1.2 Install Dependencies

```bash
pip install timm vllm==0.11.0 pandas fire
```

Install FFmpeg (required for video decoding):

```bash
conda install -c conda-forge ffmpeg -y
```

---

### 1.3 Hugging Face Login

This project downloads fine-tuned models from Hugging Face Hub.

Login with a **Fine-grained Access Token**:

```bash
huggingface-cli login
```

---

## 2. Supported Models

### 2.1 Frame-level Models (timm)

Only the following **fine-tuned timm models** are supported:

| Purpose              | Hugging Face Repository |
|----------------------|-------------------------|
| Clarity filtering | Mumulin/vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k-NTUH |
| Need / Not-Need voting | Mumulin/beit3_large_patch16_224.in22k_ft_in1k-NTUH |
|  | Mumulin/beit3_base_patch16_224.indomain_pt-NTUH |
|  | Mumulin/swinv2_large_window12_192-NTUH |

---

### 2.2 Vision-Language Models (VLM)

Users can manually choose **one** of the following VLMs for video segmentation:

- Mumulin/Qwen3-VL-4B-Instruct-NTUH
- Mumulin/Qwen3-VL-2B-Instruct-NTUH
- Mumulin/InternVL3_5-4B-HF-NTUH

---

## 3. Data Preparation

### 3.1 Case List

Create a file:

```text
configs/cases.txt
```

Each line contains **one case ID**:

```text
01072025084507_00000000U0710907
03042025105638_00000000U0429638
```
The case ID folder name under data/videos/ must exactly match the case ID listed in configs/cases.txt.

---

### 3.2 Video Directory Structure

Place videos under:

```text
data/videos/<CASE_ID>/
```

Each case directory may contain **one or multiple videos**.

Example:

```text
data/
├── videos/
│   └── 01072025084507_00000000U0710907/
│       ├── M_01072025084507_00000000U0710907_1_001-1.MP4
│       └── M_01072025084507_00000000U0710907_1_002-1.MP4
```

---

## 4. Run Pipeline

From the project root:

```bash
bash scripts/run_full_pipeline.sh
```

---

## 5. Output Structure

After running the pipeline, the project directory will look like this:

```text
CSDVideoSegment/
├── configs/
│   ├── cases.txt
│   └── phase_questions.txt
│
├── data/
│   ├── videos/
│   │   └── <CASE_ID>/
│   │       ├── *.MP4
│   │       └── ...
│   │
│   ├── frames/
│   │   └── <CASE_ID>/
│   │       └── *.png
│   │
│   └── <CASE_ID>_phase_query.json
│
├── results/
│   └── <CASE_ID>/
│       ├── clarity.csv
│       ├── clarity_filtered.txt
│       ├── need_vote.csv
│       ├── need_list.txt
│       ├── run_csv_need/
│       └── vlm/
│           └── <VLM_MODEL_NAME>/
│               └── <CASE_ID>_phase_query.jsonl
│
├── results/
│   └── <VLM_MODEL_NAME>/
│       └── <CASE_ID>_phase_query.csv   # FINAL OUTPUT
│
├── scripts/
│   └── run_full_pipeline.sh
│
└── utils/
```

All other generated directories (frames, intermediate CSVs, JSON, logs)
are internal and can be ignored.

---

## 6. Notes

- Model weights are automatically downloaded to the Hugging Face cache.
- Intermediate files are required internally but not meant for manual inspection.
- The pipeline is case-centric and supports multi-video cases.
