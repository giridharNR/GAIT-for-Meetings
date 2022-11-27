import pandas  as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from os import listdir
from os.path import isfile, join
import os
from tqdm import tqdm
import string
import re
import string

root_dir = "./output_dir/"
METRIC_OUTPUT_LOCATION =  "./metrics/metrics.csv"
SEGEMENTED_RES_OUPUT_DIR = "./segement_res"
GRAPH_DIR = "./graphs"

def get_probabilities_and_text(meeting_file):
    
    text = []
    probability = []
    
    
    for line in meeting_file:

        res = line.rsplit('\t', 1)[-1]
        try:
            data = float(res)
        except:
            data = 0.0
        text.append(line.rsplit('\t', 1)[0].strip())
        probability.append(data)
    
    return text, probability


def produce_graphs(id_, x_axis, probability):
    
    sns.set(style="whitegrid")
    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 10)
    # first plot with X and Y data
    ax1.plot(x_axis, probability, color='tab:orange', linestyle="-", label="Probability of topic end")


    # ax1.plot(df_t["Publication_count-2001"], df_t["Citation_count-2011"], color='tab:blue', linestyle="",marker=".", label="Treatment")
    ax1.tick_params(axis='both', which='major', labelsize=15)
    plt.ylabel("Probability", fontsize=20)
    plt.xlabel("Sentence", fontsize=20)
    # plt.title(title)
    plt.legend(loc="upper right", prop={'size': 18})
    plt.grid(True)
    plt.savefig('{}/graph_{}.png'.format(GRAPH_DIR, id_), bbox_inches='tight')
    # plt.show()
    
    
def process_text(text):
    
    text = text.strip()
    
    html_pattern = re.compile('<.*?>')
    text = html_pattern.sub(' ', text)
    
    
    person_reference = re.compile('\[PERSON(\d*)\]')
    text = re.sub(person_reference, r'person \1', text)
    
    person_reference = re.compile('\[PROJECT(\d*)\]')
    text = re.sub(person_reference, r'project \1', text)
    
    person_reference = re.compile('\[ORGANIZATION(\d*)\]')
    text = re.sub(person_reference, r'organization \1', text)
    
    # chars = string.punctuation.replace("(", "").replace(",", "").replace(")", "")
    chars = '<=>?@#$%&\-!~'
    text = re.sub(r'['+chars+']', '',text)
    
    text = re.sub(r'\s+', ' ', text)
    
    return text
    

def produce_segmented_text(text, probability, id_):
    
    res = ""
    
    cutoff = np.mean(probability) + np.std(probability) * 2
    segments = 0
    
    sentence_batch = []
    with open("{}/segmented-{}.txt".format(SEGEMENTED_RES_OUPUT_DIR, id_), "w") as txtf:
        
        for i in range(len(text)):

            prob = probability[i]

            sentence = text[i]

            sentence = process_text(sentence)
            
            # res+=sentence + " "
            
            sentence_batch.append(sentence)
            
            if prob >= cutoff:
                
                res+= " ".join(sentence_batch)
                sentence_batch = []
                res += "\n"
                segments += 1
        
        txtf.write(res)
   
    return segments, cutoff


def get_metric_for_meeting(text):
    
    
    data = "\n".join(text)
    
    personPattern = re.compile('(\(PERSON\d*\))')
    no_persons = len(set(personPattern.findall(data)))
    
    length_meeting_no_process = len(data.replace("\n", " ").strip().split(" "))
    
    data = process_text(data)
    length_meeting = len(data.replace("\n", " ").strip().split(" "))
    
    return no_persons, length_meeting_no_process, length_meeting
    
    
    


onlyfiles = [join(root_dir, f) for f in listdir(root_dir) if isfile(join(root_dir, f))]
# print(len(onlyfiles))

with open(METRIC_OUTPUT_LOCATION, "w") as metric_file:
    metric_file.write("id, segments, cutoff, no_persons, meeting_length(unprocessed), meeting_length(processed)\n")
    
    
    for onlyfile in tqdm(onlyfiles, total = len(onlyfiles)):
    
        id_ = onlyfile.split("/")[-1].split(".")[0]
        # print(id_)



        with open(onlyfile, "r") as meeting_file:

            text, probability = get_probabilities_and_text(meeting_file)
            segments, cutoff = produce_segmented_text(text, probability, id_)
            no_persons, length_meeting_no_process, length_meeting = get_metric_for_meeting(text)

            metric_file.write("{},{},{},{},{},{}\n".format(id_, segments, cutoff, no_persons, length_meeting_no_process, length_meeting))


            produce_graphs(id_, [x for x in range(len(text))], probability)
