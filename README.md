# Note Maker - Automated Anki Card Generator

This project automates the process of converting my personal daily notes into Anki flashcards using AI. It consists of two main components: a note maker LoRA that converts raw notes into structured JSON objects, and a deck assigner LoRA that categorizes these notes into appropriate Anki decks.

# Installation

1. clone the repo like so
`git clone --recurse-submodules="rrjr_py_pkg" https://github.com/RRomeroJr/clozer .`
2. Highly recommended to make a new virtual environment
3. Run
`python auto_setup.py`
  - You will likely see a red warning about the anki module requiring a certain version of protobuf that conflicts with some other modules. For me this hasn't been a problem yet. Will update in the future if it does.
4. Install vcpkg from microsoft to use cmake. I'm pretty sure this is only needed when converting to gguf. That is what I needed it for and I have yet to test without it.
   - `git clone https://github.com/Microsoft/vcpkg.git your\preferred\location\vcpkg`
   - `cd your\preferred\location\vcpkg`
   - `.\bootstrap-vcpkg.bat -disableMetrics`
   - `.\vcpkg install curl:x64-windows`
5. Add the following env vars to your environment. Might be different for linux/ mac.
   - `CURL_LIBRARY = "path\to\vcpkg\installed\x64-windows\lib\libcurl.lib"`
   - `CURL_INCLUDE_DIR = "path\to\vcpkg\installed\x64-windows\include"`
   - `CMAKE_PREFIX_PATH = "path\to\vcpkg\installed\x64-windows"`

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
3. Import the generated TSV file in new_notes/ into Anki

## Deck Assigner Training

### For Your Own Anki Decks setup
Skip this if you just want to test with provided example deck assigner data.

1. Go you your Anki and search/ filter for all notes in decks that you would like the deck assigner to potentially assign generated Anki notes to.
   - It's a good idea to tag your selection with something like `deck_assign_train` so that you can easily search for this tag and easily re-generate this dataset if needed.
2. Export your deck assigner dataset from Anki
3. Make sure your Anki collection path is one of the default Anki paths in the `notemaker_settings/settings.json` if not then add it
4. In `deck_assign_dataset_builder.py` scroll to `main()` should be at the bottom.
5. here you can see the call to `mk_datasets`. Some options that is has..
   - `dataset_path`: If your .tsv is not in `datasets/examples/deck-assigner/deck_assigner.tsv` so can set it here
   - `tsv_overwrite`: If True mk_datasets will add the `#test_guids:` line to your .tsv overwritting the original. If not it will try to make file called `yourTsvName_rewrite.tsv` in the same dir as the original
   - `regen_if_found`: The mk_datasets will regen the test_guids even if it found them in your .tsv
   - More documentation to be added to the `mk_datasets` method directly soon! 
6. run
```bash
python deck_assign_dataset_builder.py
```
this will generate a `deck_assigner_train.parquet` and `deck_assigner_test.parquet` file in same dir as your .tsv. Default `datasets/examples/deck_assigner`
7. in `deck_assigner_trainer.py` make sure `train` and `test` are set properly in the following code block. In case you changed the default names.
```python
dataset = load_dataset('parquet',
            data_files={
                'train': 'datasets/examples/deck_assigner/deck_assigner_train.parquet',
                'test': 'datasets/examples/deck_assigner/deck_assigner_train.parquet'
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
- Both the deck assigner and note maker are trained from the Mistral-7B-v0.1

### Output Format

The final TSV output is formatted for direct import into Anki, containing:
- Notetype
- Deck assignment
- All field data (Besides tags. Tags not supported yet)

### Requirements

- Python 3.11+ (Warning. python 3.13 could break some things.)
- PyTorch
- Transformers library
- Unsloth
- Additional dependencies listed in requirements.txt
