# CSDVideoSegment

CSDVideoSegment is an end-to-end pipeline for **case-based endoscopy video or image phase segmentation**.

---

## 🚀 PhasePro MAX (Web UI Editor)
We have developed a professional timeline editor for refining machine-generated segments. It features elastic timeline logic, multi-video support, and project history persistence.

**For detailed setup and usage instructions of the Web Application, please refer to:**
### 👉 [**Frontend & Web Editor Guide**](./frontend/README.md)

---

The pipeline automatically performs:

* Video → frame extraction (**video pipeline only**)
* Direct image ingestion (**image pipeline only**)
* Frame-level filtering and voting using fine-tuned *timm* models
* Phase inference using Vision-Language Models (VLM) via **vLLM**
* Final **phase-level CSV report generation**

Users only need to prepare **case-organized videos or images**.
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

Install FFmpeg (**required for video pipeline only**):

```bash
conda install -c conda-forge ffmpeg -y
```

---

### 1.3 Hugging Face Login

This project downloads fine-tuned models from the Hugging Face Hub.

Login with a **fine-grained access token**:

```bash
huggingface-cli login
```

---

## 2. Supported Models

### 2.1 Frame-level Models (timm)

Only the following **fine-tuned timm models** are supported and are automatically downloaded:

| Purpose                | Hugging Face Repository                                               |
| ---------------------- | --------------------------------------------------------------------- |
| Clarity filtering      | Mumulin/vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k-NTUH |
| Need / Not-Need voting | Mumulin/beit3_large_patch16_224.in22k_ft_in1k-NTUH                    |
|                        | Mumulin/beit3_base_patch16_224.indomain_pt-NTUH                       |
|                        | Mumulin/swinv2_large_window12_192-NTUH                                |

---

### 2.2 Vision-Language Models (VLM)

Users must choose **one** VLM model for phase inference:

* Mumulin/Qwen3-VL-4B-Instruct-NTUH
* Mumulin/Qwen3-VL-2B-Instruct-NTUH
* Mumulin/InternVL3_5-4B-HF-NTUH
* Mumulin/InternVL3_5-2B-HF-NTUH

The selected VLM is used identically for both video and image pipelines.

---

## 3. Data Preparation

CSDVideoSegment supports **two equivalent input modes**:

* **Video-based pipeline** (raw endoscopy videos)
* **Image-based pipeline** (pre-extracted frames)

Both pipelines are **case-centric** and produce identical outputs.

---

### 3.1 Case List (Video Pipeline Only)

Create the following file:

```text
configs/cases.txt
```

Each line contains **one case ID**:

```text
01072025084507_00000000U0710907
03042025105638_00000000U0429638
```

Each case ID **must exactly match** the corresponding folder name under `data/videos/`.

> **Note**: The image pipeline does **not** require `cases.txt`.

---

### 3.2 Video Directory Structure (Video Pipeline)

Place videos under:

```text
data/videos/<CASE_ID>/
```

Each case directory may contain **one or multiple video files**.

Example:

```text
data/
├── videos/
│   └── 01072025084507_00000000U0710907/
│       ├── M_01072025084507_00000000U0710907_1_001-1.MP4
│       └── M_01072025084507_00000000U0710907_1_002-1.MP4
```

---

### 3.3 Image Directory Structure (Image Pipeline)

Images should be placed under:

```text
data/images/<CASE_ID>/
```

* Each folder represents **one case**
* The folder name is used as the **case ID** in the output
* Images may be `.png`, `.jpg`, or `.jpeg`

Example:

```text
data/
├── images/
│   ├── frame_000001.png
│   ├── frame_000002.png
│   └── ...
```

---

## 4. Video Pipeline

The video pipeline performs **video decoding + frame sampling** before phase inference.

From the project root:

```bash
bash scripts/run_video_pipeline.sh
```

Internally, the pipeline:

1. Reads case IDs from `configs/cases.txt`
2. Decodes videos under `data/videos/<CASE_ID>/`
3. Samples frames at the configured **FPS**
4. Applies frame-level filtering and voting
5. Runs VLM-based phase inference
6. Merges predictions into a final phase-level CSV

---

## 5. Image Pipeline

The image pipeline skips video decoding and directly processes pre-extracted images.

From the project root:

```bash
bash scripts/run_image_pipeline.sh
```

Internally, the pipeline:

1. Scans all subfolders under `data/images/`
2. Treats each subfolder as an independent case
3. Applies the same frame-level filtering and voting
4. Runs the same VLM-based phase inference
5. Generates a final phase-level CSV per case

The image and video pipelines share **identical downstream logic**.

---

## 6. Output Structure

After running either pipeline, the directory structure will look like:

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
│   ├── images/
│   │   ├── *.png / *.jpg
│   │   └── ...
│   │
│   └── frames/
│       └── <CASE_ID>/
│           └── *.png
│
├── results/
│   └── <VLM_MODEL_NAME>/
│       ├── <CASE_ID>_phase_query.csv   # FINAL OUTPUT
│
├── scripts/
│   ├── run_video_pipeline.sh
│   └── run_image_pipeline.sh
│
└── utils/
```

Only the files under `results/` are intended for user consumption.

---

## 7. Configuration Notes

* **FPS** controls frame sampling rate during video decoding
* **MIN_DURATION** defines the minimum temporal length (in seconds) for a valid phase segment
* Both parameters affect **final phase aggregation**, not raw frame predictions

---

## 8. Additional Notes

* Model weights are automatically cached by Hugging Face
* Intermediate files (frames, CSVs, JSONs, logs) are internal and can be ignored
* Multi-video cases are fully supported
* Video and image pipelines are designed to be **functionally equivalent**
