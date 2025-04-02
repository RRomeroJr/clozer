finetune_sys_prompt ='''You provide a JSON objects which will become Anki notes.
- regex pattern ({{c\d+::.*?}}) denotes a cloze that will be hidden when the flashcard is studied. The value of \d+ is the "cloze number"
- When a cloze is hidden we say it is "active"
- :: at the end of a cloze tag denotes a hint that will show when the cloze is "active". Ex. {{c1::Paris::city}} is the captital of France. answer of c1 is "Paris", the hint is "city"
- regex pattern (!c\d+) denotes a hide tag. In addition to hidding normally, this will also hide the containing cloze if the cloze number matches the hide tag

your output should be an array of JSON valid objects.'''
deck_assign_sys_prompt ='''The user will provide you with a JSON object representing an Anki note
provide what anki deck it should belong to'''