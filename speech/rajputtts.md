# RajputTTS – Lightweight, Open-Weight Voice AI

**Tagline:**  
> A fast, finetune-friendly, open-weight text-to-speech engine that combines the lightweight performance of Kokoro/KittenTTS with the easy cloning and customization of Chatterbox — minus the DX headaches.

---

## Core Goals
- **Lightweight & Fast**  
  - 50–80M parameter acoustic model (FastPitch-lite / StyleTTS2-lite) + small ISTFTNet vocoder  
  - Real-time on consumer GPUs, near-real-time on CPU
- **Open-Weight & Open-Source**  
  - Apache-2.0 or similar license for model/code  
  - Model card with intended use, safety notes
- **Easy Voice Cloning & Finetuning**  
  - Few-shot speaker encoder (ECAPA-TDNN) for cloning  
  - One-command finetune on new datasets
- **Robust Pronunciation**  
  - Hybrid G2P: Misaki with `--transformer-first` option, fallback to phonemizer (eSpeak-NG)
- **Production-Ready DX**  
  - No hardcoded progress spam, configurable reporters  
  - CLI + Python API + Hugging Face integration

---

## Feature Spec

### 1. Frontend (G2P)
- **Default**: Misaki hybrid (rules/lookup first, neural fallback)  
- **Option**: Transformer-first G2P with rules fallback  
- **Fallback**: Phonemizer + eSpeak-NG for OOD words  
- Mapped to consistent phoneme inventory for acoustic model

### 2. Acoustic Model
- **FastPitch-lite** (duration + pitch + energy predictors)  
  - Output: mel spectrogram + duration info (timestamps)  
  - Configurable dimensions, layers, dropout  
- Optional: StyleTTS2-lite (decoder-only) variant

### 3. Vocoder
- **Default**: ISTFTNet-small (fast, good quality)  
- **Alt**: HiFi-GAN-small (selectable in config)

### 4. Voice Cloning
- ECAPA-TDNN speaker encoder  
- Few-shot conditioning (5–10 sec of ref audio)  
- Multi-speaker training support

### 5. Training & Finetuning
- PyTorch Lightning or Accelerate-based trainer  
- Resume-safe checkpoints (model/opt/sched/ema/scaler/state)  
- Single-GPU, DDP, and FSDP support  
- Mixed precision (fp16/bf16)  
- Configurable logging (W&B, JSONL, Rich, None)

### 6. Inference
**CLI:**
```bash
rajputtts --text "Hello world" --voice-ref ref.wav --out out.wav
```

**Python API:**
```python
from rajputtts import TTS
tts = TTS.from_pretrained("rajputtts-50m")
audio = tts.speak("Namaste", ref_audio="me.wav")
```
- Outputs audio + optional JSON timestamps

### 7. Export
- One-command Hugging Face Hub push  
- Auto-generate model card from template

---

## Roadmap

**v0.1 – MVP (Weeks 1–2)**  
- FastPitch-lite + ISTFTNet-small  
- Misaki G2P default (rules-first)  
- CLI + Python API basic inference  
- Train on LJSpeech subset → intelligible audio

**v0.2 – Multi-Voice & Cloning (Weeks 3–4)**  
- ECAPA speaker encoder integration  
- Few-shot cloning from ref audio  
- Hugging Face model export  
- Add `--transformer-first` G2P flag

**v0.3 – DX & Robustness (Weeks 5–6)**  
- Configurable progress/logging system  
- JSON timestamps from duration predictor  
- Test suite + CI  
- Data cleaning & augmentation scripts

**v0.4 – Multilingual & Docs (Weeks 7–8)**  
- Misaki extended for additional languages  
- Multilingual finetune recipes  
- Full documentation + tutorials
