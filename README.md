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

# Usage

## Personal Notes -> Importable TSV For Anki

### For Your Own Personal Unstructured Notes
Skip this if you just want to test with the provided example personal note.

Look at the exmaple personal note for reference. I spseparte info "blocks" with at least one "-" on a new line. The moodule is trained with random sparator langths and with "-", "_", "=" or "–" as separator characters. It should figure it out.

### Workflow
1. Set `todays_notes_path`in `get_today_csv.py` to the path of the personal note that you want to make Anki notes for.
2. Run the script:
```bash
python get_today_csv.py
```
3. Import the generated CSV file in new_notes/ into Anki

## Deck Assigner Training

### For Your Own Anki Decks setup
Skip this if you just want to test with provided example deck assigner data.

1. Go you your Anki and search/ filter for all notes in decks that you would like the deck assigner to potentially assign generated Anki notes to.
  - It's a good idea to tag your selection with something like `deck_assign_train` so that you can easily search for this tag and easily re-generate this dataset if needed.
2. Export your deck assigner dataset from Anki
3. In `deck_assign_dataset_builder.py` set the `dataset_path`. (Make sure your Anki collection path is one of the default Anki paths in the `notemaker_settings/settings.json` if not then add it)
4. Set `g_guids` to `True`.
  - This is going to look through your Anki export and give you some Anki note guids that you can use as your test/ eval dataset
5. Set `g_guids` to `False`.
6. run
```bash
python deck_assign_dataset_builder.py
```
this will generate a `deck_assigner_train.parquet` and `deck_assigner_test.parquet` file in `datasets/parquet`
7. in `deck_assigner_trainer.py` make sure `train` and `test` are set properly in the following code block. In case you changed the default names.
```python
dataset = load_dataset('parquet',
            data_files={
                'train': 'attempt_3_5_train.parquet',
                'test': 'attempt_3_5_test.parquet'
            },
            split=None)
```
### Workflow
1. If all your doing is training the deck assigner with the provided example data then all that is needed is to run:
```bash
python deck_assigner_trainer.py
```

This will generate a `outputs/deck_assigner` and start saving checkpoints there as the model trains

## Notemaker Training

### For Your Own Personal Anki Note Types
This gets a little more complex to set up. If all you want to is test with the provided data you can skip this section.

### Workflow
### Models
- Both are trained from the Mistral-7B-v0.1

### Note Maker Model
- Converts raw notes into structured JSON objects

### Deck Assigner Model
- Looks at the generated assigns them to an existing Anki deck

### Output Format

The final CSV output is formatted for direct import into Anki, containing:
- Notetype
- Deck assignment
- All field data (Besides tags. Tags not supported yet)

### Requirements

- Python 3.11+ (Warning. python 3.13 could break some things.)
- PyTorch
- Transformers library
- Unsloth
- Additional dependencies listed in requirements.txt
