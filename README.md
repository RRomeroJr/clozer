# Note Maker - Automated Anki Card Generator

This project automates the process of converting my daily notes into Anki flashcards using AI models. It consists of two main components: a note maker LoRA that converts raw notes into structured JSON objects, and a deck assigner LoRA that categorizes these notes into appropriate Anki decks.

## Overview

The main workflow is handled by `get_today_csv.py`, which:
1. Reads your daily note file
2. Processes it through the note maker model to create JSON objects representing Anki notes
3. Passes these JSON objects to the deck assigner model for categorization
4. Outputs a CSV file that can be directly imported into Anki

## Project Structure

```
.
├── get_today_csv.py     # Main script for processing notes
├── finetune.py          # Model fine-tuning script
├── data_prep.py         # Data preparation utilities
├── visualize.py         # Training visualization tools
├── datasets/            # Training data directory
    └── parquet/         # Processed training data
└── new_notes/           # Output dir for the generated CSV files to import to Anki
```

## Usage

1. Prepare your daily note file in your preferred format
2. Run the conversion script:
```bash
python get_today_csv.py
```
3. Import the generated CSV file in new_notes/ into Anki

## Models
- Both are trained from the Mistral-7B-v0.1

### Note Maker Model
- Converts raw notes into structured JSON objects

### Deck Assigner Model
- Looks at the generated assigns them to an existing Anki deck

## Output Format

The final CSV output is formatted for direct import into Anki, containing:
- Notetype
- Deck assignment
- All field data (Besides tags. Tags not supported yet)

## Requirements

- Python 3.11+ (Warning. python 3.13 could break some things.)
- PyTorch
- Transformers library
- Unsloth
- Additional dependencies listed in requirements.txt
