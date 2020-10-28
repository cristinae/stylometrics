
#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Calculation of several stylistic measures for an input text/collection
    Mesures include: text statistics, text semantic richness and complexity
    Measures are thought for simple text characterisation; text classification
    would require more measures (to be implemented)
    TODO: prepare a nice package
    Date: 21.10.2020
    Last modified: 25.10.2020
    Author: cristinae
"""

import sys, warnings, os
import argparse

import numpy as np
import nltk
import pyphen
import re
import string
from collections import Counter
from math import log2
from numpy import linalg as LA
import networkx as nx



def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iFile',
                    required=True,
                    type=str,
                    default="", 
                    help="Input file" )
    parser.add_argument('--stopWordFile',
                    required=False,
                    type=str,
                    default="../resources/sw/es.sw", 
                    help="File with the list of function word for the language" )
    parser.add_argument('--v', 
		    action='store_true', 
                    help="Prints the output in vertical format")
    parser.add_argument('--h', 
		    action='store_true', 
                    help="Prints the output in horizontal format")
    return parser



def printResults(results, position):
    if (position is 'h'):
       print(results)
    else:
       for key, value in results.items():
           #print(key.ljust(30, ' ') + " & " + str(value))  
           print( " & " + str(value).ljust(10, ' ') + "\\")  


def removeSWList(document, stopwordsFile):
    '''
    Removes all stopwords appearing in stopwordsFile from a list of sentences
    The output is lowercased
    '''
    fn = os.path.join(os.path.dirname(__file__), stopwordsFile)
    with open(fn) as f:
        functionWord = f.readlines()
    for i in range(len(functionWord)):
        functionWord[i] = functionWord[i].strip()    
    cleanDocument = []
    for sentence in document:
        cleanDocument.append(" ".join([w.lower() for w in sentence.split() if not w.lower() in functionWord]))
    return cleanDocument


# Text statistics; character level; scope: collection
def digit_pctg(text):
    """
    Counts the percentage of digits in a string
    """
    try:
       return sum(c.isdigit() for c in text)/len(text)
    except ZeroDivisionError:
       return 0

def uppercase_pctg(text):
    """
    Counts the percentage of uppercase words in a string
    """
    try:
       return sum(c.isupper() for c in text)/len(text)
    except ZeroDivisionError:
       return 0

def punctuation_pctg(text):
    """
    Counts punctuation marks (all together)
    """
    punct = ['\'', ':', ',', '_', '¡', '¿', '!', '?', ';', ".", '\"', '(', ')', '-']
    pCount = 0
    for i in text: 
       if i in punct: 
          pCount += 1
    try:
       return pCount/len(text)
    except ZeroDivisionError:
       return 0


# Text statistics; word level; scope: collection
def functionWords_freq(text, stopwordsFile): 
    """
    Count stopwords (all together).
    Text must be tokenised and no \n can be used to join sentences
    """
    words = text.split()
    fn = os.path.join(os.path.dirname(__file__), stopwordsFile)
    with open(fn) as f:
        functionWord = f.readlines()
    for i in range(len(functionWord)):
        functionWord[i] = functionWord[i].strip('\n')
    swCounts = 0
    for s in words:
      if s.lower() in functionWord:
        swCounts += 1
    try:
       return swCounts/len(words)
    except ZeroDivisionError:
       return 0


def wordLength(text): 
    """
    Average word length
    Text must be tokenised and no \n can be used to join sentences
    Think about removing puntuation
    """
    words = text.split()
    length = 0
    for w in words:
        length = length + len(w)
    try:
       return length/len(words)
    except ZeroDivisionError:
       return 0


def shortWords(text): 
    """
    Percentage of short words (< 4 chars)
    Text must be tokenised and no \n can be used to join sentences
    Think about removing puntuation
    """
    words = text.split()
    shortCounts = 0
    for w in words:
      if len(w) < 4:
        shortCounts += 1
    try:
       return shortCounts/len(words)
    except ZeroDivisionError:
       return 0


# Text statistics/richness; word level; scope: collection
def numWords(document): 
    """
    Number of words in a sentence (string) or a document (array of sentences)
    """
    wordCounts = 0
    if (type(document)==str):
        words = document.split()
        wordCounts = len(words)
    else:
        for sentence in document:
            wordCounts = wordCounts + len(sentence.split())
    return wordCounts


def numTypes(document): 
    """
    Number of unique words, types, in a sentence (string) or a document (array of sentences)
    """
    wordCounts = 0
    if (type(document)==str):
        words = document.split()
        wordCounts = len(set(words))
    else:
        for sentence in document:
            wordCounts = wordCounts + len(set(sentence.split()))
    return wordCounts


def hapaxLegomena_ratio(text): 
    """
    Count hapax legomena, words that appear once in a text (single string)
    (Depends on the text length)
    Text must be tokenised and lowercased
    Remove punctuation
    """
    words = text.split()
    fdist = nltk.FreqDist()
    for w in words:
        fdist[w] += 1
    fdistHapax = nltk.FreqDist.hapaxes(fdist)
    try:
       return len(fdistHapax)/len(words)
    except ZeroDivisionError:
       return 0


def hapaxDislegomena_ratio(text): 
    """
    Count hapax dislegomena, words that appear twice in a text (single string)
    (Depends on the text length)
    Text must be tokenised and lowercased/truecased if needed (German?)
    Remove punctuation
    """
    words = text.split()
    fdist = nltk.FreqDist()
    for w in words:
        fdist[w] += 1
    dislegomena = [item for item in fdist if fdist[item] == 2]
    try:
       return len(dislegomena)/len(words)
    except ZeroDivisionError:
       return 0


def ttr(text):
    """
    Type token ratio (TTR) in a text (single string)
    (Depends on the text length)
    Text must be tokenised and lowercased/truecased if needed (German?)
    Remove punctuation
    """
    words = text.split()
    types = len(set(words))
    tokens = len(words)
    try:
       return types/tokens
    except ZeroDivisionError:
       return 0


def yule_K(text): 
    """
    Yule's K measure
    Measure the vocabulary richness of a text: the larger Yule’s K, the less rich the vocabulary is. 
    (Independent of the text length)
    There are 2 implementations because I cannot reproduce an example
    https://quanteda.io/reference/textstat_lexdiv.html#examples
    """
    word_list = text.split()
    #word_list = words(text)
    token_counter = Counter(tok.lower() for tok in word_list)
    m1 = sum(token_counter.values())
    m2 = sum([freq ** 2 for freq in token_counter.values()])
    yuleI = (m1*m1) / (m2-m1)
    yuleK = 10000/yuleI
    return yuleK
    
def yuleK(text):
    '''
    Yule's K measure
    Measure the vocabulary richness of a text: the larger Yule’s K, the less rich the vocabulary is. 
    (Independent of the text length)
    K = 10^4 * (sum(fX*X^2) - N) / N^2
       where N is the number of tokens, X is a vector with the frequencies of each type, and fX is the frequencies for each X. 
    There are 2 implementations because I cannot reproduce an example
    https://quanteda.io/reference/textstat_lexdiv.html#examples
    '''
    word_list = text.split()
    freqs = Counter(tok.lower() for tok in word_list)
    freqsSpectrum = Counter(freqs.values())
    s1 = len(word_list)
    s2 = 0
    for mTimes, numWords in freqsSpectrum.items():
        s2 = s2 + mTimes*mTimes * numWords
    #s2 = sum([m * freq ** 2 for m, freq in freqsSpectrum.items()])
    try:
       yuleK = 10000 * (s2-s1) / (s1*s1)
    except ZeroDivisionError:
       yuleK = 0
    return yuleK

    
def shannonEntropy(text):
    '''
    Entropy of the distribution
    '''
    word_list = text.split()
    freqs = Counter(tok.lower() for tok in word_list)
    numWords = sum(freqs.values()) 
    try:
       return sum(freq/numWords * log2(numWords/freq) for freq in freqs.values())
    except ZeroDivisionError:
       return 0


# Complexity measures word level; scope: collection
# the variable 'document' is an array of sentences
def sentenceLength(document): 
    """
    Average sentence length
    """
    wordCounts = 0
    for sentence in document:
        words = sentence.split()
        wordCounts = wordCounts + len(words)
    try:
       return wordCounts/len(document)
    except ZeroDivisionError:
       return 0


def numSyllables(text, lan):
    """
    Number of syllables in a text
    """
    p = pyphen.Pyphen(lang=lan)
    sylCounts = 0
    if (type(text)==str):
        words = text.split()
        for w in words:
            #print(str(p.inserted(w)))
            sylCounts = sylCounts + len(p.positions(w))+1    
    else:
        for sentence in text:
            words = sentence.split()
            for w in words:
                sylCounts = sylCounts + len(p.positions(w))+1  
    return sylCounts


def wordsMoreXSyls(text, x, lan):
    """
    Number of words with more than X syllables in a text
    """
    p = pyphen.Pyphen(lang=lan)
    wordCounts = 0
    if (type(text)==str):
        words = text.split()
        for w in words:
            #print(p.inserted(w))
            if(len(p.positions(w))+1>x):
               wordCounts += 1
    else:
        for sentence in text:
            words = sentence.split()
            for w in words:
                if(len(p.positions(w))+1>x):
                   wordCounts += 1
    return wordCounts


def countHyphenatedWords(text):
    """
    Number of words with a hyphen
    """
    wordCounts = 0
    pattern = '[a-z]+-[a-z]+'
    if (type(text)==str):
        if(len(re.findall(pattern, text)) > 0):
           wordCounts = len(re.findall(pattern, text))
    else:
        for sentence in text:
            if(len(re.findall(pattern, sentence)) > 0):
               wordCounts += len(re.findall(pattern,sentence))
    return wordCounts


def kMostFreqWords(text, k):
    """
    Most frequent words in an array of sentences or a string
    """
    words = []
    if (type(text)==str):
       words = text.lower().split()
    else:
       words = [i.lower() for item in text for i in item.split()]
    return Counter(words).most_common(k)


def gunningFog(document, lan): 
    """
    Readability score for texts in English
    Gunning fog (1952):
    0.4 [(words/sentences) + 100 (complex words / words)] 
    TODO: complex words needs some PoS
    """
    complexWords = wordsMoreXSyls(document, 2, lan) - countHyphenatedWords(document)
    try:
       return (0.4*(sentenceLength(document) + 100*complexWords/numWords(document)))
    except ZeroDivisionError:
       return 0


def flesch_reading_ease(document, lan): 
    """
    Flesch Reading Ease score for texts in English
    Higher scores indicate material that is easier to read; 
    lower numbers mark passages that are more difficult to read.
    206.835 - 1.015(words/sentences) - 84.6(syllables/words)
    """
    try:
       return (206.835 - 1.015*sentenceLength(document) - 84.6*numSyllables(document,lan)/numWords(document))
    except ZeroDivisionError:
       return 0


def fernandezHuerta_ease(document, lan): 
    """
    Spanish adaptation of the Flesch Reading Ease score for analyzing texts in English
    206.84 - (0.6 × Total Number of Syllables) - 1.02 × (Total Number of Words)
    The original formula is for texts with 100 words, genaralisation used
    https://linguistlist.org/issues/22/22-2332/
    Still, I think there is a mistake and should only differ on the weights with the original Flesh
    Coco, L., Colina, S., Atcherson, S. R., & Marrone, N. (2017). 
    Readability Level of Spanish-Language Patient-Reported Outcome Measures in Audiology and Otolaryngology. 
    American journal of audiology, 26(3), 309–317. https://doi.org/10.1044/2017_AJA-17-0018
    """
    try:
       return (206.84 - (60*numSyllables(document,lan)/numWords(document)) - 1.02*sentenceLength(document))
    except ZeroDivisionError:
       return 0


def fleschSzigriszt(document, lan): 
    """
    Flesch-Szigriszt (1993)
    Readability measure for Spanish, escala INFLESZ
    perspicuidad del texto: la perspicuidad es la cualidad de perspicuo, es decir, 
    escrito con estilo inteligible (que puede ser entendido).

    IFSZ = 206.84 - (62.3 x Syllables/Words) - Words/Sentences)
    """
    try:
       return (206.84 - 62.3*numSyllables(document,lan)/numWords(document) - sentenceLength(document))
    except ZeroDivisionError:
       return 0


# Complexity measures, graph-based; scope: collection
# the variable 'document' is an array of sentences
# def adjG(document,W):
def adjG(document):
    '''
    Building a graph from words in text, only for adjacent words, coocurrence words
    # W: list(zip(*Counter(tok.lower()).most_common()))[0], all words ranked by frequency 
    '''
    G = nx.Graph()
    for element in document:
        sentence = element.split()
        # filters out all the words that are not in the frequency list. Why? all of them should be there
        #print(sentence)
        #['mejoraremos', 'teleasistencia', 'coordinación', 'policial', 'judicial', 'protección', 'mujeres', 'maltratadas']
        #sentence=list(filter(lambda x: x in W, sentence)) 
        #print(sentence)
        #['mejoraremos', 'teleasistencia', 'coordinación', 'policial', 'judicial', 'protección', 'mujeres', 'maltratadas']
        if len(sentence)>1:
            pairs=list(zip(sentence,sentence[1:]))
            #print(pairs)
            #[('mejoraremos', 'teleasistencia'), ('teleasistencia', 'coordinación'), ('coordinación', 'policial'), ('policial', 'judicial'), ('judicial', 'protección'), ('protección', 'mujeres'), ('mujeres', 'maltratadas')]
            for pair in pairs:
                if G.has_edge(pair[0],pair[1])==False:
                    G.add_edge(pair[0],pair[1],weight=1)
                else:
                    x=G[pair[0]][pair[1]]['weight']
                    G[pair[0]][pair[1]]['weight']=x+1        
    return G


def laplacianEnergy(G):
    ''' 
    Laplacian Energy for the graph G
    normalized_laplacian_matrix(G, nodelist=None, weight="weight"):
        N = D^{-1/2} L D^{-1/2}
     where `L` is the graph Laplacian and `D` is the diagonal matrix of node degrees.
    '''
    # G = adjG(document)
    M = nx.normalized_laplacian_matrix(G,weight='weight').todense()
    eigs = LA.eigvals(M)
    # max_eig = sorted(eigs)[1]
    # Laplacian energy: sum of squares (not here!) of the eigenvalues in the Laplacian matrix
    # normalized Laplacian energy, see paper
    eigs = [np.abs(x-1) for x in eigs]   
    try:
       return sum(eigs)/float(len(eigs))
    except ZeroDivisionError:
       return 0


def clusteringGraph(G):
    '''
    average_clustering(G, nodes=None, weight=None, count_zeros=True)[source]
    The clustering coefficient for the graph is the average,
    C = \frac{1}{n}\sum_{v \in G} c_v,
    where n is the number of nodes in G.
    For weighted graphs, the clustering of a node c_v is defined as the geometric average of the subgraph edge weights
    '''
    return nx.average_clustering(G)




def main(args=None):

    # get command line arguments
    parser = get_parser()
    args = parser.parse_args(args)
 
    # read input file
    with open(args.iFile) as f:
         text = f.read().replace("\n", " ")
    with open(args.iFile) as f:
         sentences = [line.rstrip() for line in f]
    with open(args.iFile) as f:
         sentencesNoPunctDig = [line.rstrip().translate(str.maketrans('', '', string.punctuation)).translate(str.maketrans('', '', string.digits)) for line in f]
    textNoPuncDig = text.translate(str.maketrans('', '', string.punctuation))
    textNoPuncDig = textNoPuncDig.translate(str.maketrans('', '', string.digits))

    # initialise results as a dictionary
    results = {}

    # text statistics measures
    results['\# sentences'] = len(sentences)
    results['\# tokens'] = numWords(text)
    results['\# types'] = numTypes(text.lower())
    results['\% digits (dig)'] = round(digit_pctg(text)*100,2)
    results['\% uppercase'] = round(uppercase_pctg(text)*100,2)
    results['\% punctuation (punc)'] = round(punctuation_pctg(text)*100,2)
    results['\% funtion words'] = round(functionWords_freq(text.lower(), args.stopWordFile)*100,2)
    results['\# words (w/o punc,dig)'] = numWords(textNoPuncDig)
    results['sentence length'] = round(sentenceLength(sentences),2)
    results['token length'] = round(wordLength(text),2)
    results['word length'] = round(wordLength(textNoPuncDig),2)
    results['\# syllables'] = numSyllables(textNoPuncDig,'es')
    results['syllables/word'] = round(numSyllables(textNoPuncDig,'es')/numWords(textNoPuncDig),2)

    # text richness measures
    results['type/token ratio'] = round(ttr(text.lower())*100,2)
    results['type/word ratio'] = round(ttr(textNoPuncDig.lower())*100,2)
    results['\% hapax legomena'] = round(hapaxLegomena_ratio(textNoPuncDig.lower())*100,2)
    results['\% dislegomena'] = round(hapaxDislegomena_ratio(textNoPuncDig.lower())*100,2)
    results['Entropy (tokens)'] = round(shannonEntropy(text),2)
    results['Entropy (words)'] = round(shannonEntropy(textNoPuncDig.lower()),2)
    results['Yule K (tokens)'] = round(yuleK(text),2)
    results['Yule K (words)'] = round(yuleK(textNoPuncDig.lower()),2)

    # text complexity measures
    results['\% short words ($<$4chars)'] = round(shortWords(textNoPuncDig),2)
    results['\# words $>$ 2 syls'] = wordsMoreXSyls(textNoPuncDig,2,'es')
    results['Fernandez Huerta'] = round(fernandezHuerta_ease(sentencesNoPunctDig,'es'),2)
    results['Szigriszt-Pazos/INFLEZ'] = round(fleschSzigriszt(sentencesNoPunctDig,'es'),2)
    results['(tok) Fernandez Huerta'] = round(fernandezHuerta_ease(sentences,'es'),2)
    results['(tok) Szigriszt-Pazos/INFLEZ'] = round(fleschSzigriszt(sentences,'es'),2)
     
    sentencesNoSW = removeSWList(sentencesNoPunctDig,args.stopWordFile) # lowercased output!
    G = adjG(sentencesNoSW)
    results['Laplacian Energy'] = round(laplacianEnergy(G),4)
    results['Clustering'] = round(clusteringGraph(G),4)
 
    if (args.v):
       printResults(results, 'v')
    else:
       printResults(results, 'h')

    print("Most common content words:")
    print(kMostFreqWords(sentencesNoSW,11))

if __name__ == "__main__":
   main()
