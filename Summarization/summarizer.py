import sys

filename = sys.argv[1]
outfile = sys.argv[2]

f = open(filename)
sentences = f.readlines()
f.close()
out = open(outfile, 'w')

from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

for sentence in sentences:
    sentence = sentence[:-1]
    summary_text = summarizer(sentence, max_length=100, min_length=20, do_sample=False)
    text = summary_text[0].get('summary_text')
    if 'CNN.com will feature iReporter' in text:
        text = ""
    out.write(text + '\n')
