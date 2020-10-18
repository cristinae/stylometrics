#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Calculates (cosine) similarity among documents. 
    Documents are represented by embeddings estimated as (weighted)
    average of sentence embeddings
    Date: 18.10.2020
    Author: cristinae
"""

import sys, warnings, os
import argparse

import numpy as np

# Constants
# Todo: commandline or config file 
NAME_L2 = '.es'
NAME_L1 = '.bpe.ca.trad2es'
EXT_DRETA = '.dreta.emb'
EXT_BASELINE = '.baseline.emb'
EXT_ESQUERRA = '.esquerra.emb'


def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iFile',
                    required=False,
                    type=str,
                    default="", 
                    metavar="<inputFile>",
                    help="Input file (Wikipedia ID without extensionsS)" )
    parser.add_argument('-f', '--iFolder',
                    required=False,
                    type=str,
                    default="", 
                    metavar="<inputFolder>",
                    help="Input folder" )
    parser.add_argument('-t', '--type',
                    required=False,
                    type=str,
                    default="mean", 
                    metavar="<embedding type>",
                    help="Type of document embedding: mean, weight" )
    parser.add_argument('-k', '--ktop',
                    required=False,
                    type=int,
                    default=-1, 
                    help="Number of sentences to be loaded from a document (default -1, all)" )
    parser.add_argument('--commonArtsFile',
                    required=False,
                    type=str,
                    default="mean", 
                    help="File of common articles as output by Wikitailor" )
    parser.add_argument('--L1',
                    required=False,
                    type=str,
                    default="ca2es.polCat", 
                    help="Folder with the articles belonging to language L1" )
    parser.add_argument('--L2',
                    required=False,
                    type=str,
                    default="es.polCat", 
                    help="Folder with the articles belonging to language L2" )
    return parser


def loadDocVecs(doc, topSentences):
    """ 
    Loads a text file with a sentence embedding per line.
    If topSentences is set, only the first sentences are loaded
    """
    vectors = []
    with open(doc, 'r') as f:
        i = 0
        for line in f:
            vector = np.fromstring(line, dtype=np.float, sep=' ')
            vectors.append(vector)
            i = i+1
            if (i==topSentences):
                break
    return vectors
  

def docEmbMean(sentenceEmbs):
    """ 
    Calculates the embedding of a document as the average of the sentence embeddings
    """
    docMatrix = np.array(sentenceEmbs)
    docEmb = np.average(docMatrix,axis=0)
    return docEmb


def cosineSim(v1, v2):
    """ 
    Cosine similarity of two vectors
    """
    return np.dot(v1, v2) / (np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2)))


def distances(bsFile, drFile, eqFile, topSentences):
    """ 
    Quick and dirty for a single document
    """    
    vecsBS = loadDocVecs(bsFile,topSentences)
    docBS = docEmbMean(vecsBS)
    vecsDR = loadDocVecs(drFile,topSentences)
    docDR = docEmbMean(vecsDR)
    vecsEQ = loadDocVecs(eqFile,topSentences)
    docEQ = docEmbMean(vecsEQ)
    print(cosineSim(docBS,docBS))
    print(cosineSim(docBS,docEQ))
    print(cosineSim(docBS,docDR))
    print(cosineSim(docDR,docEQ))


def getDocEmb(embFile, color, topSentences):
    """ 
    Loads the file with the embeddings of a document sentence per
    sentence and calculated the document embedding.
    Currently only the mean is implemented
    """    
    colorEmbFile = embFile+color
    vecs = loadDocVecs(colorEmbFile,topSentences)
    doc = docEmbMean(vecs)
    return doc


def calculateDistances(fileArticles, l1, l2, topSentences):
    """ 
    Calculates
    """
    # Header outputfile
    header = 'WP docs, simL1(bs,dr), simL1(bs,eq), simL1(dr,eq), '+ \
             'simL2(bs,dr), simL2(bs,eq), simL2(dr,eq), '+ \
             'simL2(bs,dr)-simL1(bs,dr), simL2(bs,eq)-simL1(bs,eq)' 
    print(header+'\n')
    with open(fileArticles, 'r') as f:
        for line in f:
            line = line.strip()
            ids = line.split()
            # Do files exist?
            embFileL1 = l1+'/'+ids[0]+NAME_L1
            if not os.path.isfile(embFileL1+EXT_BASELINE):
               continue
            embFileL2 = l2+'/'+ids[2]+NAME_L2
            if not os.path.isfile(embFileL2+EXT_BASELINE):
               continue
            # for L1
            embBaseL1 = getDocEmb(embFileL1, EXT_BASELINE, topSentences)
            embDretL1 = getDocEmb(embFileL1, EXT_DRETA, topSentences)
            embEsquL1 = getDocEmb(embFileL1, EXT_ESQUERRA, topSentences)
            distancesL1 = str(np.round(cosineSim(embBaseL1,embDretL1),6))+', ' + \
                          str(np.round(cosineSim(embBaseL1,embEsquL1),6))+', ' + \
                          str(np.round(cosineSim(embDretL1,embEsquL1),6))+', ' 
            # for L2
            embBaseL2 = getDocEmb(embFileL2, EXT_BASELINE, topSentences)
            embDretL2 = getDocEmb(embFileL2, EXT_DRETA, topSentences)
            embEsquL2 = getDocEmb(embFileL2, EXT_ESQUERRA, topSentences)
            distancesL2 = str(np.round(cosineSim(embBaseL2,embDretL2),6))+', ' + \
                          str(np.round(cosineSim(embBaseL2,embEsquL2),6))+', ' + \
                          str(np.round(cosineSim(embDretL2,embEsquL2),6))+', ' 
            langsDret = np.round(cosineSim(embBaseL2,embDretL2)-cosineSim(embBaseL1,embDretL1),6)
            langsEsqu = np.round(cosineSim(embBaseL2,embEsquL2)-cosineSim(embBaseL1,embEsquL1),6)
            print(line + ', ' + distancesL1 + distancesL2 + str(langsDret) + ', ' + str(langsEsqu))


def main(args=None):

    #Get command line arguments
    parser = get_parser()
    args = parser.parse_args(args)
 
    # Locate the input files
    nameINfile = args.iFile
    bsFile = nameINfile + '.baseline.emb'
    drFile = nameINfile + '.dreta.emb'
    eqFile = nameINfile + '.esquerra.emb'

    if (args.commonArtsFile is ''):
       # Distances for a document
       distances(bsFile, drFile, eqFile, args.ktop)
    else:
       calculateDistances(args.commonArtsFile, args.L1, args.L2, args.ktop)


if __name__ == "__main__":
   main()

