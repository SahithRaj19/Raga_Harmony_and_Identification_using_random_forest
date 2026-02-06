import streamlit as st
import os
import numpy as np
import librosa
import soundfile as sf
import joblib
import pretty_midi

# -------------------------------
# Paths
# -------------------------------
PROJECT_PATH = os.getcwd()
AUDIO_DIR = os.path.join(PROJECT_PATH, "data", "audio")
MODEL_PATH = os.path.join(PROJECT_PATH, "outputs", "models", "raga_random_forest.joblib")
MIDI_DIR = os.path.join(PROJECT_PATH, "outputs", "midi")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(MIDI_DIR, exist_ok=True)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"]

clf = load_model()

# -------------------------------
# Raga Scales (Harmony Constraint)
# -------------------------------
raga_scales = {

    # --- Audava / Pentatonic ---
    "Abhogi": [0, 3, 5, 7, 10],
    "Bhupali": [0, 2, 4, 7, 9],
    "Durga": [0, 2, 4, 7, 9],
    "Dhani": [0, 3, 5, 7, 10],
    "Hamsadhwani": [0, 2, 4, 7, 11],
    "Malkauns": [0, 3, 5, 8, 10],
    "Chandrakauuns": [0, 3, 5, 8, 11],
    "Madhukauns": [0, 3, 5, 8, 11],
    "Megh": [0, 2, 5, 7, 10],

    # --- Bhairav ang ---
    "Bhairav": [0, 1, 4, 5, 7, 8, 11],
    "Ahir Bhairav": [0, 2, 4, 5, 7, 8, 10],
    "Bairagi": [0, 1, 4, 5, 7, 8, 11],
    "Bibhas": [0, 1, 4, 5, 7, 8, 11],
    "Virat Bhairav": [0, 1, 4, 6, 7, 8, 11],

    # --- Bhairavi / Asavari ang ---
    "Bhairavi": [0, 1, 3, 5, 7, 8, 10],
    "Sindhu Bhairavi": [0, 1, 3, 5, 7, 8, 10, 11],
    "Asavari": [0, 2, 3, 5, 7, 8, 10],
    "Jaunpuri": [0, 2, 3, 5, 7, 8, 10],
    "Komal Bhimpalasi": [0, 3, 5, 7, 10],

    # --- Kafi ang ---
    "Bageshree": [0, 3, 5, 7, 10],
    "Bhimpalasi": [0, 3, 5, 7, 10],
    "Darbari Kanada": [0, 2, 3, 5, 7, 9, 10],
    "Jog": [0, 3, 5, 7, 10],
    "Rageshri": [0, 2, 5, 7, 9, 10],

    # --- Kalyan ang ---
    "Yaman": [0, 2, 4, 6, 7, 9, 11],
    "Kalyan": [0, 2, 4, 6, 7, 9, 11],
    "Bihag": [0, 4, 6, 7, 9, 11],
    "Hameer": [0, 2, 4, 6, 7, 9, 11],
    "Tilak Kamod": [0, 2, 4, 7, 9, 11],
    "Kedar": [0, 2, 4, 7, 9, 11],

    # --- Todi / Marwa ang ---
    "Todi": [0, 1, 3, 6, 7, 8, 11],
    "Bilaskhani Todi": [0, 1, 3, 5, 7, 8, 10],
    "Multani": [0, 1, 3, 6, 7, 8, 11],
    "Marwa": [0, 1, 4, 6, 7, 9, 11],
    "Puriya": [0, 1, 4, 6, 7, 8, 11],
    "Puriya Dhanashree": [0, 1, 4, 6, 7, 9, 11],
    "Shree": [0, 1, 4, 6, 7, 8, 11],

    # --- Bilawal / Khamaj ang ---
    "Shuddha Bilawal": [0, 2, 4, 5, 7, 9, 11],
    "Khamaj": [0, 2, 4, 5, 7, 9, 10],
    "Desh": [0, 2, 4, 5, 7, 9, 10],
    "Tilak Kamod": [0, 2, 4, 7, 9, 11],

    # --- Malhar ang ---
    "Bahar": [0, 2, 4, 5, 7, 9, 10],
    "Brindawani Malhar": [0, 2, 5, 7, 9],
    "Gaud Malhar": [0, 2, 4, 5, 7, 9, 10],
    "Miyan Malhar": [0, 2, 4, 5, 7, 9, 10],
    "Ramdasi Malhar": [0, 2, 4, 5, 7, 9, 10],
    "Sarang Malhar": [0, 2, 5, 7, 9],
    "Sawani": [0, 2, 5, 7, 9],

    # --- Sarang / Saarang ---
    "Brindawani Saarang": [0, 2, 5, 7, 9],
    "Shuddha Sarang": [0, 2, 4, 6, 7, 9, 11],
    "Patadeep": [0, 2, 4, 5, 7, 9],

    # --- Misc / Carnatic overlap ---
    "Kirwani": [0, 2, 3, 5, 7, 8, 11],
    "Nata Bhairavi": [0, 2, 3, 5, 7, 8, 10],
    "Saraswathi": [0, 2, 4, 6, 7, 9, 11],
    "Ganapati": [0, 2, 4, 7, 9],
    "Lalit": [0, 1, 4, 6, 7, 8, 11],
    "Lalit Pancham": [0, 1, 4, 6, 7, 8, 11],
    "Lagan Gandhar": [0, 2, 3, 5, 7, 9, 10],
}


# -------------------------------
# Utility Functions
# -------------------------------
def ensure_wav(file_path):
    name, ext = os.path.splitext(file_path)
    if ext.lower() == ".wav":
        return file_path

    wav_path = name + ".wav"
    y, sr = librosa.load(file_path, sr=None)
    sf.write(wav_path, y, sr)
    return wav_path


def extract_features(wav_path, duration=45.0):
    y, sr = librosa.load(wav_path, sr=22050)
    y, _ = librosa.effects.trim(y, top_db=30)
    y = y[: int(duration * sr)]

    if len(y) < sr * 5:
        return None, None, None

    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("A1"),
        fmax=librosa.note_to_hz("C8")
    )

    valid = ~np.isnan(f0)
    if not np.any(valid):
        return None, None, None

    midi = librosa.hz_to_midi(f0[valid])
    pcs = np.mod(midi, 12)

    tonic = np.argmax(np.bincount(pcs.astype(int), minlength=12))
    rel = np.mod(pcs - tonic, 12)

    hist = np.bincount(rel.astype(int), minlength=12)
    hist = hist / np.sum(hist)

    return hist, tonic, pcs


def generate_chords(scale):
    chords = []
    for i in range(len(scale)):
        for j in range(i + 1, len(scale)):
            for k in range(j + 1, len(scale)):
                chords.append([scale[i], scale[j], scale[k]])
    return chords


def select_best_chords(chords, melody_pcs, top_k=4):
    scored = []
    for c in chords:
        score = sum(1 for p in melody_pcs if p in c)
        scored.append((c, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:top_k]]


def export_midi(chords, tonic, raga):
    safe = raga.replace(" ", "_")
    out_path = os.path.join(MIDI_DIR, f"{safe}_harmony.mid")

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)

    base = 60 + tonic
    t = 0

    for chord in chords:
        for pc in chord:
            note = pretty_midi.Note(
                velocity=80,
                pitch=base + pc,
                start=t,
                end=t + 2
            )
            inst.notes.append(note)
        t += 2

    pm.instruments.append(inst)
    pm.write(out_path)

    return out_path

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="RAGA Harmony Generator", layout="centered")

st.title("🎶 RAGA Harmony Generator")
st.caption("Raga Recognition and Raga-Constrained Harmony Generation")

uploaded = st.file_uploader("Upload an audio file (WAV or MP3)", type=["wav", "mp3"])

if uploaded:
    audio_path = os.path.join(AUDIO_DIR, uploaded.name)
    with open(audio_path, "wb") as f:
        f.write(uploaded.read())

    st.audio(audio_path)

    if st.button("Analyze & Generate Harmony"):
        with st.spinner("Analyzing audio..."):
            wav_path = ensure_wav(audio_path)
            feat, tonic, melody_pcs = extract_features(wav_path)

            if feat is None:
                st.error("Feature extraction failed. Please upload a clean melodic recording.")
                st.stop()

            raga = clf.predict(feat.reshape(1, -1))[0]
            st.success(f"Predicted Raga: **{raga}**")

            if raga not in raga_scales:
                st.warning("Harmony not defined for this raga.")
                st.stop()

            chords = generate_chords(raga_scales[raga])
            selected = select_best_chords(chords, melody_pcs)

            midi_path = export_midi(selected, tonic, raga)

            st.download_button(
                "Download Harmony MIDI",
                data=open(midi_path, "rb"),
                file_name=os.path.basename(midi_path)
            )
