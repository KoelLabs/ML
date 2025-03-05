
from misaki import ja
import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from forced_alignment.common import phoneme_to_vector 
from forced_alignment.common import group_phonemes

g2p = ja.JAG2P() 

data_dir = "./data/.repos/Kokoro-Speech-Dataset/output/metadata.csv"

def parse_dataset (dir):   
    """
        this function will parse the metadata
    """  
    # split each line in meta data by the pipe character "|" and store the first part in the "file" column and the second part in the "transcript" column, third part can be ignored
    data = open(data_dir, "r").readlines()
    data = [x.split("|") for x in data]
    SUBSTITUTE = {
        "ʦ": "ts",
        "ʨ": "tʃ",
    }
    dataset = []
    for i in range(len(data)):
        sample = data[i][1] # grab japanese text
        sample = sample.replace(" ", "").replace("。", "").replace("、", "").replace("…", "")

        # grab IPA phonemes 
        phonemes, tokens = g2p(sample)
        # clean
        phonemes = phonemes.replace(" ", "").replace("“", "").replace("``", "").replace("”", "")
        # apply substitutions
        for key, value in SUBSTITUTE.items():
            phonemes = phonemes.replace(key, value)
        phonemes = group_phonemes(phonemes)
        # make a space separated string instead of list of chars
        phonemes = " ".join(phonemes)

        # path and phonemes
        dataset.append((data[i][0], phonemes))
    return dataset 


# i tʃ i k o k a s a s ə n i l i p i k n ɔ o k i n a i t a k a i m a tʃ t 
    
#     unidetifiable = []
#     for c in phonemes: 
#         if phoneme_to_vector(c) is None: unidetifiable.append(c)

# print("unidetifiable: ", unidetifiable)


