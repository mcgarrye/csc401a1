import numpy as np
import argparse
import json
import csv
from string import punctuation

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}

BGL_DATA = {}
W_DATA = {}
ALT_IDS = []
CENTER_IDS = []
LEFT_IDS = []
RIGHT_IDS = []

with open('/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv', "r") as f:
    for line in csv.reader(f):
        if line[1]:
            BGL_DATA[line[1]] = (line[3], line[4], line[5])
    
with open('/u/cs401/Wordlists/Ratings_Warriner_et_al.csv', "r") as f:
    for line in csv.reader(f):
        if line[1]:
            W_DATA[line[1]] = (line[2], line[5], line[8])

with open('/u/cs401/A1/feats/Alt_IDs.txt', "r") as f:
    id_text = f.read()
    ALT_IDS = id_text.split('\n')

with open('/u/cs401/A1/feats/Center_IDs.txt', "r") as f:
    id_text = f.read()
    CENTER_IDS = id_text.split('\n')
    
with open('/u/cs401/A1/feats/Left_IDs.txt', "r") as f:
    id_text = f.read()
    LEFT_IDS = id_text.split('\n')
    
with open('/u/cs401/A1/feats/Right_IDs.txt', "r") as f:
    id_text = f.read()
    RIGHT_IDS = id_text.split('\n')

ALT_FEATS = np.load('/u/cs401/A1/feats/Alt_feats.dat.npy')

CENTER_FEATS = np.load('/u/cs401/A1/feats/Center_feats.dat.npy')  

LEFT_FEATS = np.load('/u/cs401/A1/feats/Left_feats.dat.npy')  

RIGHT_FEATS = np.load('/u/cs401/A1/feats/Right_feats.dat.npy')  

def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    tokens = comment.split()
    features = np.zeros(173)
    token_lengths, word_count = 0, 0
    aoa_list, img_list, fam_list = [], [], []
    val_list, aro_list, dom_list = [], [], []
    previous, double_previous = None, None
    for t in tokens:
        if t[-3:] == '_SP':
            continue
        word, _, tag = t.rpartition('/')
        # Extract features that rely on capitalization.
        if len(word) >= 3 and word.isupper():
            features[0] = features[0] + 1
        # Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
        word = word.lower()

        # Extract features that do not rely on capitalization.
        if tag == 'PRP':
            if word in FIRST_PERSON_PRONOUNS:
                features[1] = features[1] + 1
            elif word in SECOND_PERSON_PRONOUNS:
                features[2] = features[2] + 1
            elif word in THIRD_PERSON_PRONOUNS:
                features[3] = features[3] + 1
        # if coordinating conjunctions
        elif tag == 'CC':
            features[4] = features[4] + 1
        # if past-tense verbs
        elif tag == 'VBD':
            features[5] = features[5] + 1
        # if comma
        elif tag == ',':
            features[7] = features[7] + len(word)        
        # if common nouns
        elif tag == 'NN' or tag == 'NNS':
            features[9] = features[9] + 1
        # if proper nouns
        elif tag == 'NNP' or tag == 'NNPS':
            features[10] = features[10] + 1
        # if adverbs
        elif tag == 'RB' or tag == 'RBR' or tag == 'RBS':
            features[11] = features[11] + 1
        # if wh- words
        elif tag == 'WDT' or tag == 'WP' or tag == 'WP$' or tag == 'WRB':
            features[12] = features[12] + 1
            
        # if future-tense verbs
        if word == 'will' or (len(word) > 3 and word[-3:] == "'ll") or word == 'gonna' or \
        (double_previous and double_previous == 'going' and previous == 'to' and tag == 'VB'):
            features[6] = features[6] + 1
 
        punctuation_count = 0
        for i in word: 
            # checking whether the char is punctuation. 
            if i in punctuation: 
                # Printing the punctuation values  
                punctuation_count = punctuation_count + 1        
        # if multi-character punctuation
        if len(word) > 1 and len(word) != punctuation_count:
            features[8] = features[8] + 1

        # if slang
        if word in SLANG:
            features[13] = features[13] + 1
            
        # if not just punctuation add to counts
        if len(word) != punctuation_count:
            token_lengths = token_lengths + len(word)
            word_count = word_count + 1

        # if end of sentence
        if word == '.':
            features[16] = features[16] + 1
        double_previous = previous
        previous = word
        
        # if in Bristol, Gilhooly, and Logie norms
        if word in BGL_DATA.keys():
            (aoa, img, fam) = BGL_DATA[word]
            # Append to list of AoA (100-700)
            aoa_list.append(float(aoa))
            # Append to list of IMG
            img_list.append(float(img))
            # Append to list of FAM
            fam_list.append(float(fam))
            
        # if in Warringer norms
        if word in W_DATA.keys():
            (val, aro, dom) = W_DATA[word]
            # Append to list of Valence
            val_list.append(float(val))
            # Append to list of Arousal
            aro_list.append(float(aro))
            # Append to list of Dominance
            dom_list.append(float(dom))   

    # average length of sentences, in tokens
    features[14] = len(tokens) / (features[16] if features[16] > 0 else 1)
    # average length of tokens, in characters
    if word_count > 0:
        features[15] = token_lengths / word_count
    
    bgl_count = len(aoa_list)
    if bgl_count > 0:
        # Average AoA (100-700)
        features[17] = sum(aoa_list) / bgl_count
        # Average IMG
        features[18] = sum(img_list) / bgl_count
        # Average FAM
        features[19] = sum(fam_list) / bgl_count
        # Standard deviation of AoA (100-700)
        features[20] = np.std(aoa_list)
        # Standard deviation of IMG
        features[21] = np.std(img_list)
        # Standard deviation of FAM
        features[22] = np.std(fam_list)   
    
    w_count = len(val_list)
    if w_count > 0:
        # Average Valence (100-700)
        features[23] = sum(val_list) / w_count
        # Average Arousal
        features[24] = sum(aro_list) / w_count
        # Average Dominance
        features[25] = sum(dom_list) / w_count
        # Standard deviation of Valence (100-700)
        features[26] = np.std(val_list)
        # Standard deviation of Arousal
        features[27] = np.std(aro_list)
        # Standard deviation of Dominance
        features[28] = np.std(dom_list)     

    return features
    
    
def extract2(feats, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feats: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    '''    
    ids, feat_data = None, None
    if comment_class == 'Center':
        ids = CENTER_IDS
        feat_data = CENTER_FEATS
    elif comment_class == 'Alt':
        ids = ALT_IDS
        feat_data = ALT_FEATS  
    elif comment_class == 'Left':
        ids = LEFT_IDS
        feat_data = LEFT_FEATS 
    elif comment_class == 'Right':
        ids = RIGHT_IDS
        feat_data = RIGHT_FEATS     
    index = ids.index(comment_id)
    feats[29:173] = feat_data[index]
    return feats
    

def main(args):
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))
    length = len(data)
    for i in range(length):
        # Use extract1 to find the first 29 features for each data point. Add these to feats.
        feats[i, :29] = extract1(data[i]['body'])[0:29]
        # Use extract2 to copy LIWC features (features 30-173) into feats.
        comment_class = data[i]['cat']
        feats[i, 29:173] = extract2(np.zeros(173), comment_class, data[i]['id'])[29:173]
        if comment_class == 'Center':
            cat = 1
        elif comment_class == 'Alt':
            cat = 3
        elif comment_class == 'Left':
            cat = 0
        elif comment_class == 'Right':
            cat = 2
        feats[i, 173] = cat
    np.savez_compressed(args.output, feats)

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        

    main(args)

