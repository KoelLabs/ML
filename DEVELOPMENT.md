# Development

## Setup

### With Pyenv

0. `git clone https://github.com/KoelLabs/ML.git`
1. Install Python 3.8.10
    - [Install pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation)
    - Run `pyenv install 3.8.10`
    - Pyenv should automatically use this version in this directory. If not, run `pyenv local 3.8.10`
2. Create a virtual environment
    - Run `python -m venv ./venv` to create it
    - Run `. venv/bin/activate` when you want to activate it
        - Run `deactivate` when you want to deactivate it
    - Pro-tip: select the virtual environment in your IDE, e.g. in VSCode, click the Python version in the bottom left corner and select the virtual environment
2. Duplicate the `.env.example` file and rename it to `.env`. Fill in the necessary environment variables.
3. Run the commands in './scripts/install.sh', e.g., with `. ./scripts/install.sh`. 
    - This will install dependencies. You should always activate your virtual environment `. ./venv/bin/activate` before running any scripts. 

### With Conda

0. `git clone https://github.com/KoelLabs/ML.git`
1. Install miniconda or anaconda
    - [Install miniconda](https://docs.conda.io/en/latest/miniconda.html)
    - Or [install anaconda](https://docs.anaconda.com/anaconda/install/)
2. Create a virtual environment
    - Run `conda create --prefix ./venv python=3.8.10` to create it
    - Run `conda activate ./venv` when you want to activate it
        - Run `conda deactivate` when you want to deactivate it
    - Pro-tip: select the virtual environment in your IDE, e.g. in VSCode, click the Python version in the bottom left corner and select the virtual environment
2. Duplicate the `.env.example` file and rename it to `.env`. Fill in the necessary environment variables.
3. Run the commands in './scripts/install.sh', e.g., with `. ./scripts/install.sh`. 
    - This will install dependencies. You should always activate your virtual environment `conda activate ./venv` before running any scripts. 

### Useful Commands

- `pip freeze > requirements.txt` - Save the current environment to a requirements file
- `pip install -r requirements.txt` - Install the requirements from a file
- `python ./scripts/audio.py record ./data/test.wav` - Record audio to a file
- `python ./scripts/audio.py play ./data/alexIsConfused.wav` - Play audio from a file
- `python ./scripts/audio.py convert ./data/openai_tts.mp3 ./data/openai_tts.wav` - Convert audio from one format to another
- `python ./scripts/audio.py text "hello there" ./data/hello_tts.wav` - Synthesize audio from text for testing

## Formatting, Linting, Automated Tests and Secret Scanning

All checks are run as github actions when you push code. You can also run them manually with `. scripts/alltests.sh`.

- We use [Black](https://black.readthedocs.io/en/stable/) for formatting. It is recommended you [integrate it with your IDE](https://black.readthedocs.io/en/stable/integrations/editors.html) to run on save. You can run it manually with `black .`. We do not enforce these styles for notebooks.

- We scan the repo for leaked secrets with [gitleaks](https://github.com/gitleaks/gitleaks). You can run it manually with `gitleaks detect`.

- We use [zizmor](https://woodruffw.github.io/zizmor/) for static analysis and security audits of github actions. You can run it manually with `zizmor .`.

## Directory Structure

```
ML/
├── .github/                     # GitHub actions and issue templates
├── data/                        # Small samples of test data
├── notebooks/                   # Interactive python notebooks
├── .data/                       # Large datasets and other hidden data
├── models/                      # Trained models organized in subfolders by third-party source
├── repos/                       # Git submodules for third-party repositories
├── scripts/                     # Shell+Python scripts
│   ├── asr/                     # Test scripts for automatic speech recognition
│   ├── eval_tests/              # Test scripts for evaluation metrics
│   ├── ipa_transcription/       # Test scripts for IPA transcription
│   ├── ipa_synthesis/           # Test scripts for IPA synthesis
│   ├── translitlat/             # Test scripts for transliteration and translation
│   ├── intonation_labeling/     # Test scripts for intonation labeling (ToBI, etc.)
│   ├── stress_detection/        # Test scripts for stress detection
│   ├── cadence_analysis/        # Test scripts for cadence analysis
│   ├── tonal_labeling/          # Test scripts for tonal labeling (e.g. Mandarin tones)
│   ├── forced_alignment/        # Test scripts for forced alignment
│   ├── voice_cloning/           # Test scripts for voice cloning
│   ├── ipa.py                   # Utils for IPA conversion
│   ├── audio.py                 # Utils for converting audio formats, recording, and playing audio
│   └── install.sh               # Setup commands            
├── .env.example                 # Example environment variables
├── .gitignore                   # Git ignore rules
├── CONTRIBUTING.md              # Contributing guidelines
├── DEVELOPMENT.md               # Development setup instructions
├── LICENSE                      # License information
├── README.md                    # Readme
└── requirements.txt             # Python dependencies
```
