# Smart Campus Security System — Multimodal Machine Learning

A multimodal AI pipeline that identifies **who** is in a restricted lab, understands **what** they are doing, and decides **whether it is authorized**, using face recognition, scene captioning, and CLIP-based activity classification fused into a final security alert.

## Overview

This project simulates a smart campus security system for restricted labs. Given a camera image of a scene, the pipeline:
1. Detects and identifies all people present using face recognition against an enrolled database.
2. Generates a natural-language caption describing the scene.
3. Classifies whether the detected activity is authorized or unauthorized using zero-shot CLIP matching against predefined activity labels.
4. Fuses identity confidence and activity status through a late-fusion decision rule to produce a final alert level (GREEN / YELLOW / RED).

## Architecture

```
Camera Image
     |
     v
+-----------------------------+
| Module 1: Face Identity     |
| InsightFace (ArcFace)       |
| -> face embeddings + cosine |
|    similarity vs enrolled DB|
+-----------------------------+
     |
     v
+-----------------------------+
| Module 2: Scene Captioning  |
| BLIP -> natural language    |
| description of the scene    |
+-----------------------------+
     |
     v
+-----------------------------+
| Module 3: Activity Check    |
| CLIP zero-shot matching     |
| vs allowed/disallowed labels|
+-----------------------------+
     |
     v
+-----------------------------+
| Module 4: Late Fusion       |
| Combine identity confidence |
| + activity status -> Alert  |
+-----------------------------+
     |
     v
Final Decision: GREEN / YELLOW / RED
```

## Modules

| Module | Model | Input | Output |
|---|---|---|---|
| 1. Face Identity | InsightFace (ArcFace) | Camera image | Person name(s) + confidence score(s) |
| 2. Scene Captioning | BLIP | Camera image | Natural language scene description |
| 3. Activity Classification | CLIP | Scene caption/image + activity labels | Authorized / Unauthorized + score |
| 4. Fusion & Alert | Weighted / rule-based late fusion | Outputs of Modules 1-3 | Final alert level + message |

## How it works

**Face Identity (Module 1):** Faces from an enrolled database (5+ team members) are embedded using InsightFace's ArcFace model. Each detected face in a test image is compared to the enrolled embeddings using cosine similarity to determine identity and confidence.

**Scene Captioning (Module 2):** BLIP generates a free-text caption describing the scene (e.g. "a group of students sitting at desks in a classroom").

**Activity Classification (Module 3):** CLIP scores the scene against a set of predefined allowed activities (e.g. working on a computer, group discussion) and disallowed activities (e.g. eating in the lab, sleeping at the desk, unauthorized photography) to flag violations.

**Fusion & Alert (Module 4):** A rule-based late fusion combines identity confidence with the activity status:
- **GREEN** — known person(s), authorized activity.
- **YELLOW** — known person but uncertain confidence, or a minor policy violation (e.g. sleeping).
- **RED** — unknown/unauthorized person detected, or a serious violation (e.g. eating in the lab).

## Results

The pipeline was tested end-to-end on 10 campus scenario images (`sce1.png`–`sce10.png`) with an enrolled database of 5 identities (Ashwath, Alice, Bob, Charlie, Ana). Example output:

| Image | Identities Detected | Scene Caption | Detected Violation | Alert Level |
|---|---|---|---|---|
| sce1.png | Charlie, Alice, Bob | a group of students sitting at desks in a classroom | a person sleeping at the desk | YELLOW |
| sce4.png | Ashwath, Bob, Unknown, Unknown | a group of people sitting around a table | a person eating food in the lab | RED |
| sce8.png | Alice, Ana, Ashwath, Charlie, Bob | a group of students sitting at desks in a classroom | None | GREEN |
| sce10.png | Unknown x6 | students in a classroom | a person eating food in the lab | RED |

The full results table, including per-identity confidence scores and CLIP activity scores, is available in the notebook.

## Tech Stack

- **Python** (Jupyter Notebook)
- **InsightFace (ArcFace)** — face detection and embedding for identity verification
- **BLIP** — image captioning / scene understanding
- **CLIP** — zero-shot activity classification via image-text similarity
- **OpenCV / PIL** — image loading and preprocessing
- **NumPy / Pandas** — scoring, similarity computation, and results tabulation
- **Matplotlib** — visualization of CLIP activity scores

## Repository Structure

```
smart-campus-security/
|-- notebook/
|   |-- Smart_Campus_Security_MML.ipynb   # Main pipeline notebook
|-- enrolled_faces/                        # Enrolled reference photos (5 identities)
|-- test_images/                           # Test scenario images (sce1.png - sce10.png)
|-- README.md
```

## How to Run

1. Clone the repository:
   ```
   git clone https://github.com/<your-username>/smart-campus-security.git
   cd smart-campus-security
   ```
2. Install dependencies:
   ```
   pip install insightface opencv-python transformers torch pandas matplotlib
   ```
3. Place enrolled reference photos in `enrolled_faces/` and test scenario images in `test_images/`. Add your picture in the enrolled faces or even replace with your classroom and classmates
4. Open and run `notebook/Smart_Campus_Security_MML.ipynb` end-to-end.

## Key Learnings

- Applied **multimodal fusion** (visual + text + biometric) to combine independent model outputs into one actionable decision.
- Implemented **zero-shot classification** with CLIP without any activity-specific training data.
- Explored failure modes of face recognition under low light/occlusion and how low identity confidence should trigger conservative (RED) alerts even when the scene caption looks benign.

## Author

**Ashwath Baskar**
