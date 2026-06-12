# app4SkinCAT
app4SkinCAT
# Rasch CAT App for the Dermatology Physician Specialist Examination in Taiwan

This repository provides a web-based Rasch computerized adaptive testing (Rasch-CAT) application for previewing, reviewing, and practicing past questions from the Taiwan Dermatology Physician Specialist Examination.

The application was developed with **Python** and **Flask**. It supports adaptive CAT, traditional non-CAT practice, bilingual item display, voice-based item reading, quick demo CAT, and CAT versus non-CAT comparison.

## Features

* Rasch-based computerized adaptive testing
* Bilingual item presentation: Chinese and English
* CAT item selection by maximum Rasch information
* Expected A Posteriori ability estimation
* Posterior standard error stopping rule
* Traditional non-CAT fixed-order practice mode
* Voice practice mode using server-generated MP3 or browser text-to-speech
* Quick 20-item CAT demo with randomized answers
* CAT versus non-CAT comparison
* CAT versus non-CAT(n) comparison using sampled examinees
* KIDMAP diagnostic dashboard
* Ability estimate, posterior SE, percentile, INFIT, and OUTFIT output
* Response history and item-selection path
* Optional item images or reference figures
* Deployable to Google App Engine

## Repository Structure

```text
.
├── app.py                         # Flask Rasch-CAT application
├── replay_bundle.zip              # Required RaschOnline replay bundle
├── requirements.txt               # Python dependencies
├── app.yaml                       # Google App Engine deployment file
├── pic/                           # Optional local image/audio folder
└── README.md
```

If your main file is named differently, for example `raschcatskin.py`, either rename it to `app.py` or adjust the deployment command accordingly.

## Required Replay Bundle

The application expects a file named:

```text
replay_bundle.zip
```

This file must be placed in the same folder as `app.py`.

The bundle should contain the following files:

```text
response_category.csv
fixed_item_delta.csv
person_estimates.csv
item_estimates.csv
metadata.json
```

Optional files include:

```text
original_response.csv
readme.md
pic/
```

## Required CSV Structure

### `response_category.csv`

This file stores the item content and answer key.

Required columns:

| Column  | Description                                         |
| ------- | --------------------------------------------------- |
| `key`   | Correct answer key, such as A, B, C, or D           |
| `no`    | Item number                                         |
| `link`  | Optional image, figure, graph path, or external URL |
| `item`  | Chinese item text and options                       |
| `item2` | English item text and options                       |

Example:

```csv
key,no,link,item,item2
A,1,pic/item001.png,"中文題幹...(A)...(B)...","English stem... A. ... B. ..."
```

### `fixed_item_delta.csv`

This file stores Rasch item difficulties.

Typical columns:

```text
ITEM, DELTA
```

The application extracts item numbers from the `ITEM` field and merges them with `response_category.csv`.

### `person_estimates.csv`

This file is used to estimate the prior mean and standard deviation for ability estimation.

### `item_estimates.csv`

This file is used for item-fit display and Wright Map visualization.

### `original_response.csv`

This optional file contains simulated or real examinee response data. It is used in CAT versus non-CAT comparison modes.

## Installation

Create and activate a Python virtual environment.

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install flask numpy pandas gtts gunicorn
```

A minimal `requirements.txt` may be:

```text
Flask
numpy
pandas
gTTS
gunicorn
```

## Running Locally

Make sure the following files are in the same folder:

```text
app.py
replay_bundle.zip
```

Then run:

```bash
python app.py
```

Alternatively, use Flask directly:

```bash
flask --app app run --host=0.0.0.0 --port=8080
```

Open the app in a browser:

```text
http://127.0.0.1:8080
```

## Application Modes

### 1. CAT Mode

CAT mode selects the next item adaptively according to the examinee’s current ability estimate.

Main settings:

* Maximum CAT items
* Stop CAT when posterior SE is less than or equal to the target value
* Starting theta
* Language: Chinese or English

After each response, the application updates:

* Ability estimate, theta
* Posterior standard error
* Item-selection path
* Response-level residual information

The test stops when the target SE is reached, the maximum number of items is reached, or all available items have been administered.

### 2. non-CAT Mode

non-CAT mode presents items in fixed item-number order. Users can select the starting item number. After the last item, the sequence wraps back to Item 1, allowing users to review the complete item bank.

This mode is suitable for:

* Full item browsing
* Traditional question-bank practice
* Focused review
* Checking all past examination questions

### 3. Voice Practice Mode

Voice practice mode randomly samples items within a selected theta range and reads the items aloud.

It supports:

* Chinese or English reading
* Stem and option reading
* Pause before the correct answer
* Correct-answer reading
* Auto-next item function
* Mobile-friendly server-generated MP3 audio
* Browser text-to-speech fallback

This mode is useful for reducing visual burden and helping users become familiar with question stems and answer options.

### 4. Quick 20-Item CAT Demo

This mode runs a fixed 20-item adaptive CAT demonstration. The system randomly selects answers and then displays the CAT result page immediately.

This mode is useful for:

* Testing the app workflow
* Demonstrating CAT item selection
* Showing KIDMAP and result dashboards quickly

### 5. CAT versus non-CAT Comparison

This mode compares CAT and full non-CAT performance using response data from `original_response.csv` when available.

Outputs include:

* CAT item length
* Full non-CAT item length
* CAT theta
* Full non-CAT theta
* Box plots
* t-test summaries
* KIDMAP and CPC results

### 6. CAT versus non-CAT(n) Comparison

This mode samples multiple examinees from `original_response.csv` and compares CAT with full non-CAT across the selected persons.

This mode is useful for evaluating:

* CAT efficiency
* Item reduction
* Person-measure consistency
* Measurement precision under SE-based stopping

## Output Dashboard

After testing, the result page displays:

* Final theta
* Posterior SE
* Percentile
* Number of administered items
* Stop reason
* INFIT MNSQ
* OUTFIT MNSQ
* Response history
* CAT trend chart
* KIDMAP dashboard
* Category probability curve
* Bank item-fit plot

The examinee view does not display the hidden answer key or per-item score during CAT and non-CAT testing.

## KIDMAP Interpretation

The KIDMAP dashboard visualizes response patterns against the final ability estimate. It helps identify:

* Items answered correctly despite being difficult
* Items answered incorrectly despite being easy
* Standardized residual patterns
* Potential misfit or unexpected responses

This supports diagnostic feedback rather than simple score reporting.

## Image and Figure Support

The `link` column in `response_category.csv` may contain:

* External URLs
* Local image filenames
* Local paths such as `pic/item001.png`

Supported image formats include:

```text
.png
.jpg
.jpeg
.gif
.webp
.svg
```

Local images can be stored either:

```text
pic/
```

beside `app.py`, or inside:

```text
replay_bundle.zip/pic/
```

## Security Note

The app uses a Flask session secret key. For deployment, set the environment variable:

```bash
CAT_SECRET_KEY=your-secure-secret-key
```

Do not use the default demo secret key in production.

## Google App Engine Deployment

A basic `app.yaml` file can be written as:

```yaml
runtime: python312

entrypoint: gunicorn -b :$PORT app:app

env_variables:
  CAT_SECRET_KEY: "replace-with-a-secure-secret-key"
```

Deploy with:

```bash
gcloud app deploy
```

Open the deployed app:

```bash
gcloud app browse
```

## Suggested `requirements.txt`

```text
Flask
numpy
pandas
gTTS
gunicorn
```

## Suggested `.gitignore`

```text
__pycache__/
*.pyc
.venv/
.env
.DS_Store
raschcatskin_tts_cache/
```

If the item bank or examination content is restricted, do not upload `replay_bundle.zip` publicly. Instead, provide instructions for authorized users to place the bundle beside `app.py`.

## Limitations

This application is intended as a learning-support and proof-of-concept Rasch-CAT platform. If item difficulties and response data are generated through simulation, results should be interpreted as evidence of system feasibility rather than empirical validation.

Before using the system for high-stakes assessment, future studies should use real examinee response data to evaluate:

* Item calibration accuracy
* Test validity
* Reliability
* Item exposure rate
* CAT stopping rules
* CAT versus non-CAT measurement agreement

## Citation

When using or modifying this application, please cite the related study or repository:

```text
Chien TW, Shao Y, Chou W, Wang WC. RaschOnline app for Rasch rating scale model. 2026.
```

Repository:

```text
https://github.com/smilechien/raschcat/
```

## License

Please specify a license before public release. Suggested options include:

* MIT License for open-source reuse
* CC BY-NC for noncommercial educational use
* Custom restricted license if examination items are copyrighted or confidential

## Contact

For questions, suggestions, or collaboration, please contact the repository maintainer.
