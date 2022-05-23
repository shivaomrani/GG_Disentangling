#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 22:27:21 2020

@author: atoney
"""

import numpy as np
import pickle
import pandas as pd
import random
from scipy.stats import norm
import math
from scipy.spatial.distance import cosine
import multiprocessing as mp


def findDeviation(array):
    array.sort()
    mean1 = np.mean(array)
    squareSum = 0

    for i in range(len(array)):
        squareSum += ((array[i] - mean1) ** 2)
    dev = math.sqrt((squareSum) / (len(array) - 1))
    return dev

def calculateCumulativeProbability(arr, value):
    cumulative=-100
    arr.sort()
    cumulative = norm.cdf(value, np.mean(arr), findDeviation(arr))
    return cumulative
def getNullDistribution(idx, nullMatrix, setSize, iterations):
    distribution = [0]*iterations

    row = list(nullMatrix[idx])
    for itr in range(iterations):
        np.random.shuffle(row)
        break_length = int(setSize / 2)
        meanFirstAttribute = np.mean(row[0:break_length])
        meanSecondAttribute = np.mean(row[break_length:setSize])
        distribution[itr] = meanFirstAttribute - meanSecondAttribute
    return distribution

def getWordEmbedding(wordEmbeddingFile, word):
    return wordEmbeddingFile[word]

def effectSize(array, mean):
    d = findDeviation(array)
    es = mean / d
    return es

def removeCategoryWordsIfNotInDictionary(vocab_array, semanticModel):
    corpus_words = list(semanticModel.keys())
    remove_words = []
    for wd in vocab_array:
        if wd not in corpus_words:
            print(wd)
            remove_words.append(wd)
    new_vocab_array = [x for x in vocab_array if x not in remove_words]
    return new_vocab_array

def cs_sim(x, y):
    return 1 - cosine(x, y)

VOCABULARY = None
BOTH_STEREOTYPES = None
TO_SHUFFLE = None
CONCEPT1_NULL_MATRIX = None
SEMANTIC_MODEL = None
MEAN_CONCEPT1_STEREOTYPE1 = None
MEAN_CONCEPT1_STEREOTYPE2 = None
ITERATIONS = None

def computeEffectSizeAndPVal(i):
    nullDistributionConcept1 = [0] * len(BOTH_STEREOTYPES)

    for itr in range(len(BOTH_STEREOTYPES)):
        col_val = TO_SHUFFLE[itr]
        item = CONCEPT1_NULL_MATRIX[i][col_val]

        nullDistributionConcept1[itr] = item

    nullDistribution = getNullDistribution(i, CONCEPT1_NULL_MATRIX,
                                           len(BOTH_STEREOTYPES), ITERATIONS)

    delta = MEAN_CONCEPT1_STEREOTYPE1[i] - MEAN_CONCEPT1_STEREOTYPE2[i]
    e = effectSize(nullDistributionConcept1, delta)
    p = 1-calculateCumulativeProbability(nullDistribution, delta)
    # print(i, "Vocabulary Word: ", VOCABULARY[i], "Effect Size: ", e, "P Val: ", p)
    return (e, p)


def WordEmbeddingFactualAssociationTestVocab(semanticModel, vocabToTest, lang):
    vocabulary = vocabToTest
    iterations = 10000
    wordDimension = 200

    if lang == 'es':

        pleasant = ["caricia", "libertad", "salud", "amor", "diploma", "paz", "placer", "ánimo", "amigo", "cielo",
                "leal", "diamante", "delicado", "honesto", "afortunado", "arco-iris", "obsequio", "honor", "milagro",
                 "amanecer", "familia", "feliz"]
        unpleasant = ["maltrato", "choque", "inmundicia", "asesinato", "enfermedad", "accidente", "muerte", "pena", "ponzoña", "hedor",
                   "asalto", "desastre", "odio", "contaminar", "tragedia", "divorcio", "cárcel", "pobreza", "feo", "cáncer", "matar",
                   "podrido", "vómito", "agonía", "prisión"]
    elif lang == 'it':
        pleasant = ["carezza", "libertà", "salute", "amore", "pace", "allegria", "amico", "paradiso", "leale", "piacere",
                "diamante", "gentile", "onesto", "fortunato", "arcobaleno", "diploma", "regalo", "onore", "miracolo",
                "alba", "famiglia", "felice", "risate", "paradiso", "vacanza"]
        unpleasant = ["abuso", "schianto", "sporcizia", "omicidio", "malattia", "incidente", "morte", "dolore", "veleno",
                  "puzza", "assalto", "disastro", "odio", "inquinare", "tragedia", "divorzio", "prigione", "povertà",
                  "brutto", "cancro", "marcio", "uccidere", "vomito", "agonia", "prigione"]
    elif lang == "fr":

        pleasant = ["caresse", "liberté", "santé", "amour", "paix", "acclamation", "ami", "paradis", "fidèle", "plaisir",
                    "diamant", "doux", "honnête", "chanceux", "arc en ciel", "diplôme", "cadeau", "honneur", "miracle",
                    "lever du soleil", "famille", "heureux", "rire", "paradis", "vacances"]
        unpleasant = ["abus", "accident", "crasse", "meurtre", "maladie", "accident", "mort", "chagrin", "poison",
                    "puanteur", "agression", "désastre", "haine", "polluer", "tragédie", "divorce", "prison", "pauvreté",
                    "laid", "cancer", "tuer", "pourri", "vomir", "agonie", "prison"]
    elif lang == "de":

        pleasant = ["liebkosung", "freiheit", "gesundheit", "liebe", "frieden", "jubel", "freund",
                "himmel", "treue", "vergnügen", "diamant", "sanft", "ehrlich", "glücklich",
                "regenbogen", "diplom", "geschenk", "ehre", "wunder", "sonnenaufgang", "familie",
                "glücklich", "lachen", "paradies", "urlaub"]
        unpleasant = ["missbrauch", "absturz", "schmutz", "mord", "krankheit", "unfall", "tod", "trauer",
                      "gift", "gestank", "angriff", "katastrophe", "hass", "umweltverschmutzung", "tragödie",
                      "scheidung", "gefängnis", "armut", "hässlich", "krebs", "töten", "faul", "erbrechen", "qual",
                       "das Gefängnis"]
    elif lang == 'pl':
        pleasant = ["pieszczota", "swoboda", "zdrowie", "miłość", "dyplom", "pokój", "przyjemność", "dopingować",
                    "przyjaciel", "niebiosa", "wierny", "diament", "delikatny", "uczciwy", "fartowny", "tęcza",
                    "podarunek", "honor", "cud", "rodzina", "szczęśliwy", "śmiech", "raj", "wakacje", "świt"]
        unpleasant = ["nadużycie", "wypadek", "brud", "zabójstwo", "choroba", "awaria", "śmierć", "smutek", "trucizna",
                          "smród,atak", "katastrofa", "nienawiść", "zanieczyszczać", "tragedia", "rozwód", "więzienie",
                          "bieda", "brzydki", "rak", "zgniły", "wymiociny", "agonia", "areszt", "zło"]
    elif lang == "tr":

        pleasant = ["okşamak", "özgürlük", "sağlık", "sevgi", "barış", "neşe", "arkadaş", "cennet" , "sadık", "keyif", "pırlanta",
                    "kibar", "dürüst", "şanslı", "gökkuşağı" ,"diploma", "hediye", "onur", "mucize", "gündoğumu", "aile", "mutlu",
                    "kahkaha","cennet", "tatil"]

        unpleasant = ["istismar", "çarpmak" ,"pislik", "cinayet", "hastalık", "ölüm", "üzüntü", "zehir",
                      " kokuşmuş ", "saldırı", "felaket", "nefret", "kirletmek", "facia", "boşanmak",
                      "hapishane", "fakirlik", "çirkin", "kanser", "öldürmek", "çürümüş", "kusmuk",
                      "ızdırap", "sancı", "cezaevi"]
    elif lang == "fa":
        pleasant = ['نوازش', 'آزادی', 'سلامتی', 'عشق', 'صلح', 'تشویق کردن', 'دوست', 'بهشت', 'وفادار', 'لذت', 'الماس', 'ملایم', 'صادقانه', 'خوش شانس', 'رنگین کمان', 'دیپلم', 'هدیه', 'افتخار و احترام', 'معجزه', 'طلوع خورشید', 'خانواده', 'خوشحال', 'خنده', 'بهشت', 'تعطیلات']
        unpleasant = ['سو استفاده', 'تصادف', 'کثافت', 'قتل', 'بیماری', 'تصادف', 'مرگ', 'غم', 'سم', 'بدبو', 'حمله', 'فاجعه', 'نفرت', 'آلوده', 'فاجعه', 'طلاق', 'زندان', 'فقر', 'زشت', 'سرطان', 'کشتن', 'پوسیده', 'استفراغ', 'عذاب', 'زندان']

    pleasant1 = removeCategoryWordsIfNotInDictionary(pleasant, semanticModel)
    unpleasant1 = removeCategoryWordsIfNotInDictionary(unpleasant, semanticModel)

    attributesFirstSet = pleasant1
    attributesSecondSet = unpleasant1

    if len(pleasant1) != len(unpleasant1):
        pleasant = random.shuffle(pleasant1)
        unpleasant = random.shuffle(unpleasant1)
        diff = abs(len(pleasant1) - len(unpleasant1))
        new_len = 25 - diff
        attributesFirstSet = pleasant1[0:new_len]
        attributesSecondSet = unpleasant1[0:new_len]


    vocabulary = removeCategoryWordsIfNotInDictionary(vocabulary, semanticModel)


    meanConcept1Stereotype1 = [0] * len(vocabulary)
    meanConcept1Stereotype2 = [0] * len(vocabulary)


    bothStereotypes = attributesFirstSet + attributesSecondSet
    random.shuffle(bothStereotypes)

    toShuffle = [i for i in range(len(bothStereotypes))]
    random.shuffle(toShuffle)

    random.shuffle(attributesFirstSet)

    random.shuffle(attributesSecondSet)


    #vocab to attributeFirstSet
    for i in range(len(vocabulary)):
        concept1Embedding = getWordEmbedding(semanticModel, vocabulary[i])

        for j in range(len(attributesFirstSet)):
            stereotype1Embedding = getWordEmbedding(semanticModel, attributesFirstSet[j])
            similarityCompatible = cs_sim(concept1Embedding, stereotype1Embedding)
            meanConcept1Stereotype1[i] += similarityCompatible
        meanConcept1Stereotype1[i] /= (len(attributesFirstSet))
    #print(meanConcept1Stereotype1)

    #vocab to attributeSecondSet
    for i in range(len(vocabulary)):
        concept1Embedding = getWordEmbedding(semanticModel, vocabulary[i])

        for j in range(len(attributesSecondSet)):
            stereotype2Embedding = getWordEmbedding(semanticModel, attributesSecondSet[j])
            similarityCompatible = cs_sim(concept1Embedding, stereotype2Embedding)
            meanConcept1Stereotype2[i] += similarityCompatible
        meanConcept1Stereotype2[i] /= (len(attributesSecondSet))

    #print(meanConcept1Stereotype2)


    concept1NullMatrix = np.zeros((len(vocabulary), len(bothStereotypes)))



    for i in range(len(vocabulary)):
        concept1Embedding = getWordEmbedding(semanticModel, vocabulary[i])
        for j in range(len(bothStereotypes)):
            nullEmbedding = getWordEmbedding(semanticModel, bothStereotypes[j])
            similarityCompatible = cs_sim(concept1Embedding, nullEmbedding)
            concept1NullMatrix[i][j]=similarityCompatible

    global VOCABULARY
    VOCABULARY = vocabulary
    global BOTH_STEREOTYPES
    BOTH_STEREOTYPES = bothStereotypes
    global TO_SHUFFLE
    TO_SHUFFLE = toShuffle
    global CONCEPT1_NULL_MATRIX
    CONCEPT1_NULL_MATRIX = concept1NullMatrix
    global SEMANTIC_MODEL
    SEMANTIC_MODEL = semanticModel
    global MEAN_CONCEPT1_STEREOTYPE1
    MEAN_CONCEPT1_STEREOTYPE1 = meanConcept1Stereotype1
    global MEAN_CONCEPT1_STEREOTYPE2
    MEAN_CONCEPT1_STEREOTYPE2 = meanConcept1Stereotype2
    global ITERATIONS
    ITERATIONS = iterations

    # print("starting multiprocessing")

    output = []
    for i in range(len(vocabulary)):
        out = computeEffectSizeAndPVal(i)
        output.append(out)

    effect_size_vector = list(map(lambda x: x[0], output))
    pval_vector = list(map(lambda x: x[1], output))

    WEFAT_Results = pd.DataFrame()
    WEFAT_Results["word"] = vocabulary
    WEFAT_Results["effect_size"] = list(effect_size_vector)
    WEFAT_Results["p_value"] = list(pval_vector)

    return WEFAT_Results
