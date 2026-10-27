# Multilingual NLU System — Intent Classification + Slot Filling

Streamlit app for **intent classification** and **slot filling** across 51 languages, built with XLM-RoBERTa and trained on the MASSIVE dataset.

## Features

- **Intent classification**: 60+ intents (alarm_set, play_music, weather_query, etc.)
- **Slot filling**: 40+ slot types (time, place_name, song_name, etc.)
- **51 languages** via zero-shot cross-lingual transfer
- **Streamlit UI**: Enter an utterance and see intent + extracted slots

## Local setup

1. **Clone and enter the repo**
   ```bash
   git clone https://github.com/AdithyaReddyGeeda/Intent-classification-and-slot-filling.git
   cd Intent-classification-and-slot-filling
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Add the trained models** (not in the repo due to size):
   - **Intent model**: Place your saved checkpoint in `results_intent_classification/checkpoint-71965/` (or `checkpoint-57572/`). The app expects a Hugging Face `AutoModelForSequenceClassification` checkpoint (e.g. XLM-RoBERTa) with `config.json` containing `id2label`.
   - **Slot model**: Place your saved model in `slot_filling_model_crf/final_model/` with tokenizer files and `id2label.json`.

4. **Run the app**
   ```bash
   streamlit run app.py
   ```
   Open http://localhost:8501.

## Deploy on Streamlit Cloud

1. **Push this repo to GitHub** (code only; models are in `.gitignore`).

2. **Host the models** (required for the app to run):
   - **Option A**: Upload the model folders to [Hugging Face Hub](https://huggingface.co/) (as a private or public model repo), then in `app.py` replace the local paths with `from_pretrained("your-username/your-intent-model")` and `from_pretrained("your-username/your-slot-model")`.
   - **Option B**: Use [Streamlit’s “Secrets”](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management) to point to a URL or path where the app can download the model files at startup (you’d add a small download step in `app.py`).

3. **Deploy**:
   - Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and select this repo.
   - Set **Main file path** to `app.py`.
   - Add any secrets (e.g. HF token) if you use private models.
   - Deploy. The first run may be slow while dependencies and models load.

## Project layout

- `app.py` — Streamlit UI and inference (intent + slot filling)
- `requirements.txt` — Python dependencies
- `results_intent_classification/` — Intent model checkpoints (add locally; gitignored)
- `slot_filling_model_crf/final_model/` — Slot filling model (add locally; gitignored)
- `*.ipynb` — Notebooks for training and evaluation

## Requirements

- Python 3.8+
- PyTorch, Transformers, Streamlit (see `requirements.txt`)

