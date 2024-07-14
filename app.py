# Loading all necessary libraries
import streamlit as st
import spacy
from collections import Counter
import random
import nltk
from nltk.corpus import wordnet as wn , stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import re

## downloading : 
nltk.download('wordnet')
nltk.download('stopwords')
#from PyPDF2 import PdfReader


st.markdown("# NLP Project :")
st.write("The purpose of this project is to generate an MCQ or a summary based on an uploaded file both in french or in english")

#def read_pdf(file): 
  #  text =""
  #  reader = PdfReader(file)
  #  for num in range(len(reader.pages)):
       # page= reader.pages[num].extract_text()
       # text += page
   # return text*/
## QCM :
def read_txt(file):
    text=""
    text = file.read().decode('utf-8')
    return text

def read_file(file,type):
    if type == "txt":
        return read_txt(file)
    if type == "Pdf": return read_pdf(file)

def map2list (map):
    list = []
    for i in range(len(map)):
        list.append(map[i])
    return list

def get_synonyms(word):
    synonyms = []
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms

def get_distractors_wordnet(syn,word):
    distractors=[]
    word= word.lower()
    orig_word = word
    if len(word.split())>0:
        word = word.replace(" ","_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0: 
        return distractors
    for item in hypernym[0].hyponyms():
        name = item.lemmas()[0].name()
        if name == orig_word:
            continue
        name = name.replace("_"," ")
        name = " ".join(w.capitalize() for w in name.split())
        if name is not None and name not in distractors:
            distractors.append(name)
    return distractors

def generate_MCQ(text, nlp, num = 5):
    if text is None:
        return []
    doc = nlp(text)
    ques =[sent.text for sent in doc.sents]
    if (len(ques)<num): num = len(ques)
    if (len(ques)>10): num = random.randint(num,10)

    else: num = random.randint(num,len(ques))
    selected = random.sample(ques, num)
    qcm = []
    for sent in selected : 
        doc_sent = nlp(sent)
        nouns = [sent.text for sent in doc_sent if sent.pos_ == "NOUN"]
        if len(nouns)<2: continue

        noun_counts = Counter(nouns) ## To compute the occurence of each noun 

        ## We'll use the most frequent noun to generate the mcq 
        if noun_counts:
            chosen = noun_counts.most_common(1)[0][0]
            distractors = []
            question = sent.replace(chosen,'______________')
            chosen = " ".join(w.capitalize() for w in chosen.split())
            choices = [chosen]
            syns = wn.synsets(chosen,'n')
            if (len(syns)>1):
                distractors = get_distractors_wordnet(syns[0],chosen)
            distractor = list(set(nouns) - {chosen})
            for i in range(len(distractor)):
                distractors.append(distractor[i])
           # syn = get_synonyms(chosen)
           # for i in range(len(syn)):
            #    distractors.append(syn[i])
            random.shuffle(distractors)
            for distractor in distractors[:4]:
                choices.append(distractor)

            random.shuffle(choices)
            correct_answer = chr(64 + choices.index(chosen) + 1)  # Convert index to letter
            qcm.append((question, choices, correct_answer))
    return qcm

def map_to_form(qcm):
    for i in range(len(qcm)):
        st.radio(qcm[i][0], map2list(qcm[i][1]), key=f"qcm_{i}")
    submitted = st.button("Display correct answers")  
    if submitted:
        for i in range(len(qcm)):
            st.write("Correct Answer: " + str(qcm[i][2]))


## Summarization :
def splitting (text,nlp):
     sentences = []
     doc = nlp(text)
     print('ORIGINAL TEXT:\n')
     for sentence in doc.sents:
        #print(sentence)
        sentence_str = str(sentence)
        cleaned_sentence = re.sub(r"[^a-zA-ZÀ-ÿ]", " ", sentence_str).split()
        sentences.append(cleaned_sentence) 
     return sentences

def similarity(sent_list_1, sent_list_2, stopwords=None):
    if stopwords is None:
        stopwords = []
        
    sent_list_1 = [w.lower() for w in sent_list_1]
    sent_list_2 = [w.lower() for w in sent_list_2]
 
    words = list(set(sent_list_1 + sent_list_2))
 
    vect1 = [0] * len(words)
    vect2 = [0] * len(words)
 
    for w in sent_list_1:
        if w in stopwords:
            continue
        vect1[words.index(w)] += 1
 
    for w in sent_list_2:
        if w in stopwords:
            continue
        vect2[words.index(w)] += 1

    return 1 - cosine_distance(vect1, vect2)

def build_similarity_matrix(sentences, stop_words):
    
    sim_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: 
                continue
            sim_matrix[idx1][idx2] = similarity(sentences[idx1], sentences[idx2], stop_words)
            
    return sim_matrix

def summerize(text,nlp,stopwords,n=3):
    summary = ""
    summarize_text = []
    sentences = splitting(text, nlp)
    if (len(sentences)<n ): n = 1
    similarity_matrix = build_similarity_matrix(sentences, stopwords)

    sentence_similarity_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)   
    
    for i in range(n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
    summary = ". ".join(summarize_text)
    return summary + ". "

if 'qcm' not in st.session_state:
    st.session_state.qcm = None
if 'summary' not in st.session_state:
    st.session_state.summary = None 

st.subheader("Uploading a file")
text = ""
file = st.file_uploader("Upload your file")
language = st.selectbox("Select a language:", ["Fr", "En"])
tr = st.selectbox("You want to generate:", ["MCQ", "Summary"])

if tr == "MCQ":
    num_ques = st.slider("Select number of questions:", 1, 7, 3)
else:
    num_sent = st.slider("Select number of summary sentences:", 1, 10 ,3)
submitted = st.button("Generate")

if submitted:
    if file:
        text = read_file(file,"txt")
        if language == "Fr":
            nlp = spacy.load("fr_core_news_md")
            stop_words = stopwords.words('french')
        else:
            nlp = spacy.load("en_core_web_md")
            stop_words = stopwords.words('english')
        if tr == "MCQ":
            st.session_state.qcm = generate_MCQ(text, nlp,num_ques)
        else:
            print(splitting(text,nlp))
            st.session_state.summary = summerize(text,nlp,stop_words,num_sent)
    else:
        st.write("Please upload your file")

if (st.session_state.qcm!=None):
    st.write("## QCM : ")
    map_to_form(st.session_state.qcm)
    if st.button("Clear"):
            st.session_state.qcm = None
            st.session_state.summary = None
            st.rerun()

if (st.session_state.summary!=None):
    st.write("## Summarization : ")
    st.write("### Text : ")
    st.write(text)
    st.write("### Summary : ")
    st.write(st.session_state.summary)

if st.button("Clear"):
            st.session_state.qcm = None
            st.session_state.summary = None
            st.rerun()
