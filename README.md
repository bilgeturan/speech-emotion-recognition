# Speech Emotion Recognition with a Hybrid Dataset

A deep learning system that predicts a speaker's emotion from audio, built to work not just on clean, acted test data, but on **natural, real-world speech**.

## The problem I wanted to solve

Most Speech Emotion Recognition (SER) models look great on paper because they're trained and tested on **actor-based datasets**, where emotions are exaggerated and clearly performed. The moment you feed them real conversational speech, accuracy collapses. This is the **lab-to-field gap**.

My first model had exactly this problem: it only worked when emotions were delivered theatrically, and failed on ordinary speech. Instead of chasing a higher (but misleading) score on acted data, I chose to close that gap honestly.

## What I did

- Trained an initial model on three actor-based datasets: **RAVDESS, TESS, and CREMA-D**.
- Diagnosed the generalization failure on natural speech.
- Added the **MELD** dataset (natural conversational speech from a TV series) to the training pool, expanding it from **~15,900 to ~29,600 samples**.
- Merged the low-sample, acoustically similar `calm` class into `neutral`, reducing the task from 8 to **7 emotion classes**.
- Fixed a preprocessing mismatch between training and inference (audio length / sample-rate / padding) that was silently hurting real-world predictions.
- Added **Grad-CAM explainability** so the model's decisions aren't a black box — you can see which MFCC regions it focuses on for each emotion.
- Built an interactive **Streamlit demo** where you can upload audio and see the prediction plus the Grad-CAM heatmap.

## Result

**~70% accuracy on the mixed dataset.**

This number is lower than what actor-only models report and that's the point. It reflects performance on messy, natural speech, which is a far more honest and useful measure of real-world reliability.

## Tech stack

`Python` · `TensorFlow / Keras` · `librosa` · `NumPy` · `scikit-learn` · `Streamlit`

- **Feature extraction:** MFCC
- **Model:** 1D Convolutional Neural Network (1D-CNN)
- **Explainability:** Grad-CAM

## Datasets

| Dataset | Type | Role |
|---|---|---|
| RAVDESS, TESS, CREMA-D | Actor-based | Initial training |
| MELD | Natural conversational | Closing the lab-to-field gap |

> Datasets are not included in this repo due to size. Links: [RAVDESS](https://zenodo.org/record/1188976), [TESS](https://tspace.library.utoronto.ca/handle/1807/24487), [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D), [MELD](https://affective-meld.github.io/).

## Running the demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

<!-- TODO: add a screenshot of the Streamlit app + a Grad-CAM heatmap here -->

## What I'd improve next

- Balance the class distribution further (some emotions are still underrepresented).
- Test on Turkish speech (the model is currently English-centric).
- Compare the 1D-CNN against a transformer-based audio encoder.

---

*Built as part of my deep learning coursework and later extended for a master's seminar at Ankara University.*
