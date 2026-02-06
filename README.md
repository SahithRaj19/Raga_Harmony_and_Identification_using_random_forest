# Raga_Harmony_and_Identification_using_random_forest
🎶 RAGA Harmony Generator

An end-to-end system for Indian classical raga recognition and raga-aware harmony generation using melodic pitch analysis and machine learning. The project automatically processes audio input, predicts the raga, and generates harmonically valid chords exported as MIDI.

📌 Overview

This project analyzes monophonic Indian classical music recordings to:

Identify the underlying raga

Generate raga-constrained harmonic accompaniment

Export the harmony as a MIDI file for musical exploration

The system is designed to be reproducible, interpretable, and research-oriented.

✨ Key Features

Supports MP3 and WAV audio formats

Automatic MP3 → WAV conversion

Silence trimming and audio preprocessing

Pitch-based melodic feature extraction

RandomForest-based raga classification

Persistent model storage using joblib

Rule-based raga-aware harmony generation

MIDI export compatible with DAWs

Streamlit UI for interactive demonstration

🧠 System Workflow

Audio file is uploaded (MP3/WAV).

MP3 files are converted to WAV for uniform processing.

Silence and noise are removed from the audio signal.

Melodic pitch is extracted and normalized relative to the tonic.

A trained RandomForest model predicts the raga.

Raga-specific note constraints are applied.

Harmonically valid chords are generated.

Harmony is exported as a raga-specific MIDI file.

Project File structure:
                        RAGA_Harmony/
                        │
                        ├── RAGA_Harmony_2.ipynb        # Core pipeline (training + inference)
                        ├── app.py                     # Streamlit-based UI
                        ├── requirements.txt
                        │
                        ├── data/
                        │   ├── audio/                 # Input audio files (MP3/WAV)
                        │   └── annotations/
                        │       └── raga_labels.csv
                        │
                        └── outputs/
                            ├── models/
                            │   └── raga_random_forest.joblib
                            └── midi/
                                └── <RagaName>_harmony.mid
🎹 Output

Predicted raga name

Generated raga-aware harmony

Downloadable MIDI file:

<RagaName>_harmony.mid

⚠️ Limitations

Designed for monophonic melodic audio (vocal, flute, violin)

Not suitable for polyphonic or percussion-heavy recordings

Raga representation is based on pitch-set approximations, not full gamaka modeling

📚 Technologies Used

Python

Librosa

Scikit-learn

RandomForestClassifier

Joblib

PrettyMIDI

Streamlit

👤 Authors

Primary Author: Sahith Raj
