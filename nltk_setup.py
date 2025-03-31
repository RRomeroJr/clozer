import os
custom_dir = os.path.abspath('.venv\\nltk_data')  # Change this to your preferred path

os.environ['NLTK_DATA'] = custom_dir
import nltk

nltk.download('words', download_dir=custom_dir)