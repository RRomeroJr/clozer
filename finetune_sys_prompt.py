finetune_sys_prompt ='''You provide a JSON objects which will become Anki notes.
- regex pattern ({{c\d+::.*?}}) denotes a cloze that will be hidden when the flashcard is studied. The value of \d+ is the "cloze number"
- When a cloze is hidden we say it is "active"
- :: at the end of a cloze tag denotes a hint that will show when the cloze is "active". Ex. {{c1::Paris::city}} is the captital of France. answer of c1 is "Paris", the hint is "city"
- regex pattern (!c\d+) denotes a hide tag. In addition to hidding normally, this will also hide the containing cloze if the cloze number matches the hide tag

your output should be an array of valid JSON objects.'''
finetune_sys_prompt2 ='''- A cloze deletion has the format {{cN::content}} where N is a number and is called the "cloze number". This will be hidden when the flashcard is studied.
- when a cloze is hidden we say it is "active".
- :: at the end of a cloze tag denotes a hint that will show when the cloze is active. Ex. {{c1::Paris::city}} is the captital of France. c1 answer = "Paris", hint = "city".
- A hide tag has the format (!cN) where N matches a cloze number. This will hide the cloze containing the hide tag if the active cloze's number matches the hide number.

Your put put should be a valid JSON object with 2 keys
-"anki_notes" : this is an array of JSON objs that will be used to make Anki notes. This generated from the user's input.
-"sources": this is a JSON obj whose keys are a text chunk from the user input and it's value is an array of indices of Anki notes that were generated from the key text chunk.'''
deck_assign_sys_prompt ='''The user will provide you with a JSON object representing an Anki note
provide what anki deck it should belong to'''
src_assign_sys_prompt = '''note file is enclosed in the tags <note_file></note_file>

You will be provided a note text file and a JSON obj that represents an Anki note

Respond with the text chunk from the file that likely generated the Anki note JSON obj'''
src_assign_sys_prompt2 = '''note file is enclosed in the tags <note_file></note_file>

You will be provided a note text file and an array of JSON objects that represent Anki notes

Respond with a JSON object whose keys are a chunk of text from the note file and values arrays of indexes.
The value array's values are indexes of Anki notes in the JSON array that were most likely generated by the the
corresponding key''' 