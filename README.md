# Text Content Moderator

A text moderation app that labels short text as **Appropriate** or **Inappropriate** using a fine-tuned `bert-base-uncased` model, served through a small Flask web app with a JSON response.

> **Content warning:** the dataset (`preprocessed_data.csv`) contains real tweets with hateful and offensive language.

## Features

- Binary classification (`Appropriate` / `Inappropriate`) with a probability for each class
- Browser UI for pasting text and seeing the result right away
- HTTP endpoint that returns JSON, so other apps can call it
- Cleanup for social-media text: mentions, HTML entities, URLs, punctuation, stemming, stopwords
- Notebooks for data preprocessing and model training/evaluation

## Results

Evaluated on a held-out test set of 4,957 tweets (20% of the data):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Appropriate | 0.90 | 0.84 | 0.87 | 835 |
| Inappropriate | 0.97 | 0.98 | 0.97 | 4,122 |

**Overall accuracy: 95.66%.** For the *Inappropriate* (positive) class: precision 96.72%, recall 98.11%, F1 97.41%.

## How It Works

```text
text ──► preprocess ──► BERT tokenizer ──► fine-tuned BERT ──► softmax ──► label + probabilities
```

The preprocessing in [App/main.py](App/main.py) does the same steps as the training notebooks, in this order:

1. Lowercase
2. Remove `@mentions`
3. Remove HTML entities (e.g. `&amp;`)
4. Remove URLs
5. Remove punctuation and noise symbols (`" ' ! \` . , # : ?`)
6. Apply Porter stemming
7. Remove English stopwords (plus `rt`)

## Dataset and Training

- **Source:** a hate-speech / offensive-language tweet dataset (`hate_offensive_data.csv`, not included in this repo). Tweets labelled *neither* became `0` (Appropriate). Tweets labelled *hate speech* or *offensive* became `1` (Inappropriate).
- **Prepared data:** [preprocessed_data.csv](preprocessed_data.csv) has 24,783 rows with columns `tweet`, `clean_tweet`, `label` (20,620 inappropriate / 4,163 appropriate).
- **Split:** 64% train / 16% validation / 20% test (`random_state=42`)
- **Training:** `BertForSequenceClassification` from `bert-base-uncased`, AdamW (lr `1e-5`), batch size 8, 3 epochs, on a Kaggle GPU

| Notebook | Purpose |
|---|---|
| [Data_Preprocessing.ipynb](Data_Preprocessing.ipynb) | Loads the raw dataset, maps labels, cleans text, writes `preprocessed_data.csv` |
| [BERT_final.ipynb](BERT_final.ipynb) | Fine-tunes BERT, evaluates it, saves the model to `/kaggle/working/bert_model/` |

## Project Structure

```text
text_content_moderator/
├── App/
│   ├── main.py               # Flask app: preprocessing, inference, routes
│   ├── templates/
│   │   └── index5.html       # Web UI
│   └── bert_model/           # Fine-tuned model (not in git, see Setup)
├── BERT_final.ipynb          # Training and evaluation
├── Data_Preprocessing.ipynb  # Data cleaning
├── preprocessed_data.csv     # Cleaned training data
├── PROJECT_REPORT.md         # Detailed project report
└── requirements_py313.txt    # Dependencies (Python 3.13)
```

## Setup

**1. Clone and create a virtual environment**

```bash
git clone https://github.com/Ephrem-shimels21/text_content_moderator.git
cd text_content_moderator
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements_py313.txt
```

> `main.py` still has the line `from google.cloud import vision`, but that package is commented out in the requirements. Either install it (`pip install google-cloud-vision`) or comment out that import. If you do neither, the app fails at startup with an `ImportError`.

**3. Add the trained model**

The weights are too large for git, so `App/bert_model/` is in `.gitignore`. To create it, run [BERT_final.ipynb](BERT_final.ipynb) (for example on Kaggle), download `/kaggle/working/bert_model/`, and place it at `App/bert_model/`:

```text
App/bert_model/
├── config.json
├── model.safetensors
└── tokenizer files (vocab.txt, tokenizer_config.json, special_tokens_map.json)
```

The app loads the model weights from this folder. It loads the tokenizer from the Hugging Face Hub (`bert-base-uncased`), so the first run needs an internet connection. NLTK data (`punkt`, `punkt_tab`, `stopwords`) is also downloaded at startup.

## Running the App

Run it from inside `App/`, because the model path is relative (`./bert_model/`):

```bash
cd App
python main.py
```

Then open <http://127.0.0.1:8081>, paste some text, and click **Analyze Content**.

## API

`POST /` with a form field named `text`. The endpoint expects form data, not JSON.

```bash
curl -X POST -F "text=Have a great day everyone" http://127.0.0.1:8081/
```

Example response (the probabilities shown are only illustrative):

```json
{
  "text": "Have a great day everyone",
  "predicted_class": "Appropriate",
  "probabilities": {
    "appropriate": 0.97,
    "inappropriate": 0.03
  }
}
```

If the `text` field is missing, you get `400 {"error": "Invalid request"}`.
