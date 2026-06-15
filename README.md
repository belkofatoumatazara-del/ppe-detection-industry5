# PPE Compliance Detection for Industry 5.0

A computer vision system that detects personal protective equipment (PPE) on
workers near machinery and **flags non-compliance** — i.e. workers missing a
hard hat, vest, or mask. Built for the *Computer Vision for Industry 5.0* final
project (ECE Engineering School, 2025–2026).

> **Industry 5.0 angle:** human-centric safety. The system doesn't just detect
> equipment — it identifies workers who are *not* protected, enabling real-time
> safety monitoring on the factory floor.

## Approach

Transfer learning with **YOLOv8** (Ultralytics), fine-tuned on the Roboflow
*Construction Site Safety* dataset. No training from scratch.

- **Task:** object detection (bounding boxes per worker / equipment)
- **Dataset:** Roboflow Universe — `roboflow-universe-projects/construction-site-safety` (v30, YOLOv8 format)
- **Model:** `yolov8s.pt` fine-tuned (~10 classes incl. NO-Hardhat / NO-Safety Vest / NO-Mask)
- **Prototype:** Streamlit (image upload) + OpenCV (live webcam)

## Repository structure

```
ppe-detection-industry5/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── 01_exploration.ipynb   # dataset download + exploration (start here)
├── app/                       # Streamlit / webcam prototype (added later)
├── src/                       # helper scripts (added later)
├── data/                      # dataset (gitignored — pulled from Roboflow)
├── runs/                      # YOLO training outputs (gitignored)
└── models/                    # final best.pt lives here (gitignored — see Releases)
```

## Getting started

1. Open `notebooks/01_exploration.ipynb` in Google Colab (Runtime → T4 GPU).
2. Add Roboflow API key as a Colab **secret** named `ROBOFLOW_API_KEY`
    . Never hard-code it.
3. Run all cells — the notebook downloads the dataset and produces sample
   visualizations + a class-balance analysis.

## Results

_To be filled after training: mAP@0.5, per-class AP, confusion matrix,
baseline vs fine-tuned comparison._

## Author

_Fatoumata Zara IBRAHIM BELKO_ — MSc Artificial Intelligence, ECE
Course: Computer Vision for Industry 5.0 — Dr. Yosra Hajjaji
