# Petrolog AI

**A platform for petrographic and petrological analysis based on artificial intelligence**

A platform for analyzing microscopic shelf and anshelf images, ICP-MS geochemical data, and 360° optical observations in a single multimodal system, forming a ready-made laboratory report.

🔗 **Jonli demo:** https://USERNAME.github.io/petrolog-ai/
📄 Project: President AI Award 2026 · Direction: AI in industry and business

> `USERNAME` o'rniga o'z GitHub foydalanuvchi nomingizni yozing.

---

## Problem

The volume of geological exploration in Uzbekistan is growing, but each shelf and anshelf is still characterized by hand:

- **Time** - A complete petrographic description of a single sample takes an average of 3–5 working days. At the Kh.M. Abdullaev Institute of Geology and Geophysics, 1,800 analyses were performed in 2024, and 2,000 in 2025.
- **Subjectivity** - two specialists describe the same slip differently; results are not reproducible.
- **Fragmentation** - the shelf image, anshlif, ICP-MS table, and expert opinion are never combined into a single database.

No. Solution

Data from four sources is processed on a single AI assembly line:

| Input | What AI does | Output |
|---|---|---|
| Shelf image (passed beam) | Segmentation of mineral phases, texture analysis | Mineral composition, % ratio |
| Anschlieff image (reflected light) | Recognition of ore minerals | Ore mineralogy |
| ICP-MS table | Geochemical classification, Clark coefficients | Petrogenetic interpretation |
| 360° pleochroism video | Optical property change recording | Identification confirmation |

The user chooses one of four modes: shrift description, anshrift description, ICP analysis, or **comprehensive analysis**. In integrated mode, all sources are processed together to generate a unified standardized report.

---

No. What is in this repository?

This is a **running demo prototype** of the platform ('index.html'). One file, server-free, fully in-browser.

**What the demo shows:**

1. Upload a shelf and/or anshelf image
2. Automatic phase separation - k-means clustering (RGB + 4D feature vector on local texture gradient, k-means++ initialization)
3. Match each phase with a possible mineral using an optical properties reference table and provide a **confidence level**.
4. Visualization of the phase map and calculation of the percentage content.
5. Analysis of ICP-MS data - preliminary classification according to TAS, mineralization anomalies according to A/CNK and A/NK indices, and Clark coefficients.
6. Automatic text petrographic description generation.
7. Download the report

### Demo restrictions — we speak openly

**No trained neural network is used in this step.** Segmentation is based on a clustering algorithm, and mineral names are based on a reference table.

The reason is simple: training the CNN model requires an expert-described reference dataset, which is created during the first stage of the project. The demo demonstrates **three-to-three conveyor** performance — accuracy comes with a dataset.

The main advantage of the project lies not in technology, but in **data**: anyone can write CNN's architecture, and no one else in the country has 5,000+ analysis results describing the rocks of Uzbekistan by an expert.

---

No. Road map

| Stage | Timeframe | Result | KPI |
|---|---|---|---|
| Dataset kernel | 1-3 months | Digitization of 20+ scientific reports, annotation regulations | ≥1,500 annotated images |
| First model | 4-6 months | Segmentation + classification model training | F1 ≥ 0.75 for the main rock-forming minerals |
| Closed beta | 7–9 months | Real workflow testing in the institute's laboratory | ≥300 samples conducted through the system |
| Pilot client | 10-12 months | Industrial pilot implementation | ≥1 contract; time savings ≥60% |

The final target for accuracy will be set after the results of the pilot dataset.

No. Planned technological stack

- **Model:** Python / PyTorch - segmentation (U-Net / SegFormer class), classification (CNN + confidence), transfer learning
- **Backend:** FastAPI
- **Frontend:** React
- **Data:** PostgreSQL + Object Storage
- **Shipping:** Docker (on-premise option for clients with high privacy requirements)

**Human-in-the-loop required:** a result below the confidence threshold is labeled "expert review required" and is never automatically verified. Each expert correction returns to the dataset in the next training cycle.


---

No. Launch

No installation required:

" 'bash
git clone https://github.com/USERNAME/petrolog-ai.git
cd petrologist-ai
Open index.html # in your browser
" '

Or a live demo: https://USERNAME.github.io/petrolog-ai/

Upload a micrograph of any shelf to test and enter sample data in the ICP-MS field:

```
SiO2,62.4,%
Al2O3,15.8,%
Fe2O3,5.2,%
CaO,4.1,%
Na2O,3.6,%
K2O,2.9,%
MgO,2.4,%
Cu,340,ppm
Au,0.8,ppm
```

---

No. Team

Scientific core - H.M. Abdullaev Institute of Geology and Geophysics: 1 Doctor of Science and 5 Doctors of Philosophy in Geological and Mineralogical Sciences. Experience of cooperation in the mineralogical and petrographic direction with NMMC, AMMC and JSC "Uzbekgeologiya qidiruv."

No. Strategic Framework

The project corresponds to the "Strategy for the Development of Artificial Intelligence Technologies until 2030" of the President of the Republic of Uzbekistan and the direction of the Decree No. PF-189 on the introduction of artificial intelligence in priority sectors of the economy.


---

## English summary

**Petrolog AI** is a multimodal AI platform for petrographic and petrological analysis of rocks. It combines thin-section images, polished-section images, ICP-MS geochemistry and 360° optical observation into a single analytical pipeline that produces a standardised laboratory report.

This repository contains a **working browser-based demo** (`index.html`, zero dependencies): upload a thin-section image, the app segments mineral phases via k-means clustering over RGB + local texture features, matches each phase against an optical-properties reference table with a confidence score, parses ICP-MS data for geochemical classification and ore anomalies, and generates a downloadable report.

The demo deliberately uses classical CV rather than a trained CNN — the expert-annotated reference dataset required for supervised training is the first deliverable of the project. The team's advantage is the data: 5 000+ expert-described rock analyses from 20+ completed research projects on Uzbek geology.

Submitted to **President AI Award 2026** (Uzbekistan), industry & business AI track.

---

