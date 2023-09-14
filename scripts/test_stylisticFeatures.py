#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit test for functions in stylisticFeatures.py
    Date: 25.10.2020
    Author: cristinae
"""

import unittest
import stylisticFeatures as sf

SWfile_en = '../resources/sw/en.sw'
SWfile_es = '../resources/sw/es.sw'
SWfile_es = '../resources/sw/de.sw'

class StylisticFeaturesTestCase(unittest.TestCase):

    # Text statistics; character level; scope: collection
    def testDigit_pctg(self):
        sentence = 'April 23rd 1998 at 14:53,53231'
        self.assertEqual(sf.digit_pctg(sentence),0.5,'Incorrect percentage of digits')

    def testUppercase_pctg(self):
        sentence = 'Saint JORDI is on aPril 23rd'
        self.assertEqual(sf.uppercase_pctg(sentence),0.25,'Incorrect percentage of uppercase letters')

    def testPunctuation_pctg(self):
        sentence = '¡Morning, Angela! :)'
        self.assertEqual(sf.punctuation_pctg(sentence),0.25,'Incorrect percentage of punctuation marks')

    # Text statistics; word level; scope: collection
    def testFunctionWords_freq(self):
        paragraph = "This is a counter of stopwords in English . It concatenates several sentences as large documents can be requested ." 
        self.assertEqual(sf.functionWords_freq(paragraph,SWfile_en),0.45,'Relative frequency of stopwords incorrect in English')
        paragraph = "Ahora miramos en castellano . ¿ Funciona este texto ?" 
        self.assertEqual(sf.functionWords_freq(paragraph,SWfile_es),0.2,'Relative frequency of stopwords incorrect in Spanish')

    def testWordLength(self):
        paragraph = "How many letters    have words here ?" 
        self.assertEqual(sf.wordLength(paragraph),4,'The average length of words fails')

    def testShortWords(self):
        paragraph = "How many    short words can you count ?" 
        self.assertEqual(sf.shortWords(paragraph),0.5,'Counting the average number of words with less than 4 letters fails')

    # Text statistics/richness; word level; scope: collection
    def testNumWords(self): 
        paragraph = []
        paragraph.append("How many short words can you count ?" )
        paragraph.append("How many letters have words here ?")
        self.assertEqual(sf.numWords(paragraph),15,'Counting the total number of words in an array fails')
        self.assertEqual(sf.numWords(paragraph[0]),8,'Counting the total number of words in a string fails')

    def testkMostFreqWords(self): 
        paragraph = []
        paragraph.append("How many short words can you count ?" )
        paragraph.append("How many letters have words here ?")
        self.assertCountEqual(sf.kMostFreqWords(paragraph,1),[('how',2)],'Counting the most frequent words in an array fails')
        self.assertCountEqual(sf.kMostFreqWords(paragraph[0],2),[('how',1),('many',1)],'Counting the most frequent words in a string fails')

    def testLegomena(self):
        text='in corpus linguistics a hapax legomenon is a word that occurs only once within a context ' + \
             'either in the    written record of an entire language in the works of an author or in a single text ' + \
             'the term is sometimes incorrectly used to describe a word that occurs in only one of an author ' + \
             'works but in more than once in that particular work hapax legomenon is a transliteration of Greek ' + \
             'meaning something being said only    once ' + \
             'the related terms dis legomenon tris legomenon and tetrakis legomenon respectively ' + \
             'refer to double triple or quadruple occurrences but are less commonly used'
        # 100 words, 41 hapax, 9 dis legomena
        self.assertEqual(sf.hapaxLegomena_ratio(text),0.41,'Estimated hapax legomena ratio is incorrect')
        self.assertEqual(sf.hapaxDislegomena_ratio(text),0.09,'Estimated dislegomena ratio is incorrect')


    def testTTR(self):
        text ='lexical density is a concept in computational linguistics that measures the structure and ' + \
              'complexity of human communication in a language lexical density estimates the linguistic ' + \
              'complexity in a written or spoken composition from the functional words grammatical units  ' + \
              'and content words lexical units lexemes one method to calculate the lexical density  ' + \
              'is to compute the ratio of lexical items to the     total number of words'
        # 65 words, 40 uniq (130 syllables, 38.92 Gunning Fox, -28.34 Flesch Reading)
        self.assertEqual(round(sf.ttr(text),4),0.6154,'Incorrect TTR measure')


    def testYuleK(self): 
        text_web = 'anyway like i was saying shrimp is the fruit of the sea you can  barbecue it boil it broil it bake it saute it'
        #text_web = "Anyway, like I was saying, shrimp is the fruit of the sea. You can barbecue it, boil it, broil it, bake it, saute it."
        self.assertEqual(round(sf.yuleK(text_web),2),381.94,'Incorrect Yule K constant')


    def testSentenceLength(self): 
        document = []
        document.append("How many short words can you count ?" )
        document.append("How many letters have words here ?")
        self.assertEqual(sf.sentenceLength(document),7.5,'The estimated average sentence length is wrong')


    def testNumSyllables(self):
        document = []
        document.append("Let's count the number of syllables in English" )
        document.append("It can be done for a sentence or a full document or array of sentences")
        document.append("done")
        paragraph = 'La legibilidad lingüística de un texto se puede medir aplicándole algoritmos sencillos que son específicos de cada lengua y requieren una investigación científica previa para su validación  No tiene en cuenta la tipografía o la presentación que también son importantes no las descuides El internauta tiene poca paciencia Nadie aguanta un tostón La web no se lee se escanea '
        self.assertEqual(sf.numSyllables(document,'en'),33,'The number of total syllables is incorrect for the document')
        self.assertEqual(sf.numSyllables(document[2],'en'),1,'The number of total syllables is incorrect for a single sentence')
        self.assertEqual(sf.numSyllables(paragraph,'es'),123,'The number of total syllables is incorrect for a single string in Spanish')


    def testWordsMoreXSyls(self):
        document = []
        document.append("Let's count the number of syllables in English with more than x syllables" )
        document.append("It can be done for a sentence or a full document or array of sentences")
        # sentences has 2 syls!
        self.assertEqual(sf.wordsMoreXSyls(document[1], 2, 'en'),1,'The number of total syllables is incorrect for a single sentence')
        self.assertEqual(sf.wordsMoreXSyls(document, 2, 'en'),3,'The number of total syllables is incorrect for a document')


    def testHyphenatedWords(self):
        document = []
        document.append("What about double-check, it is ok -Kerl")
        document.append("-It can be done for a sentence or a full document or array of sentences")
        # sentences has 2 syls!
        self.assertEqual(sf.countHyphenatedWords(document[0]),1,'The number of hyphenated words is incorrect for a single sentence')
        self.assertEqual(sf.countHyphenatedWords(document),1,'The number of hyphenated words is incorrect for a document')


    def testEnglishReadability(self): 
        document = []
        document.append("did you know that the average attention span for a website visitor is only seconds")
        document.append("that 's roughly how long it takes the average person to read two average sentences")
        document.append("that 's not much time to grab someone 's attention")
        document.append("the trick is to use plain language where possible")
        document.append("do n't make your reader work to get to the point")
        document.append("use short sentences rather than long sentences which drone on and on and try to say too much and your readers lose interest part way through like this one")
        # Fleish 80.56, Gunning Fog 10.43, 89 words, 117 silabes 
        #self.assertEqual(round(sf.gunningFog(document,'en'),2),10.43,'The Gunning Fog index is incorrect for a document')
        self.assertEqual(round(sf.flesch_reading_ease(document,'en'),2),80.56,'The Flesch reading-ease index is incorrect for a document')

    def testSpanishReadability(self):
        document = []
        document.append("la legibilidad lingüística de un texto se puede medir aplicándole algoritmos sencillos que son específicos de cada lengua y requieren una investigación científica previa para su validación")
        document.append("no tiene en cuenta la tipografía o la presentación que también son importantes no las descuides")
        document.append("el internauta tiene poca paciencia")
        document.append("nadie aguanta un tostón")
        document.append("la web no se lee se escanea")
        # Fernández Huerta 66.84 (vs. 69.72), Szigriszt-Pazos/INFLEZ 62.19 (vs. 65.16)
        # sílabas 129 (vs 123), palabras 59
        # Differences come because of errors in counting syllables
        self.assertEqual(round(sf.fernandezHuerta_ease(document,'es'),2),69.72,'The Fernandez Huerta reading-ease index is incorrect for a document')
        self.assertEqual(round(sf.fleschSzigriszt(document,'es'),2),65.16,'The Szigriszt-Pazos/INFLEZ reading-ease index is incorrect for a document')

    def testGermanReadability(self):
        document = []
        document.append("Alle meine Entchen schwimmen auf dem See Köpfchen unters Wasser Schwänzchen in die Höh ")
        # FREde=74
	# WSTF1 = 0,1935*0 + 0,1672*14 + 0,1297*29 − 0,0327*43 − 0,875 = 3,8
        # Differences come because of errors in counting syllables
        self.assertEqual(round(sf.flesch_reading_ease(document,'de'),0),74,'The Flesh reading-ease index is incorrect for a document')
        self.assertEqual(round(sf.wienerSTF(document,'de'),1),3.8,'Wiener Sachtextformel index is incorrect for a document')


# Let's start!
if __name__ == '__main__':
    unittest.main()

