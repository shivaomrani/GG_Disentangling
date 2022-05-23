from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm as norm1
import statistics as stat
from scipy.stats import norm
from numpy import dot
import random as random


def computeNullMatrix(names, bothStereotypes, vectors, word2index):

    concept1NullMatrix = []

    for name in names:

        concept1Embedding = vectors[word2index[name]]
        my_list = []

        for attribute in bothStereotypes:
            nullEmbedding = vectors[word2index[attribute]]
            similarityCompatible = cosineSimilarity(concept1Embedding, nullEmbedding)
            my_list.append(similarityCompatible)
        concept1NullMatrix.append(my_list)

    return concept1NullMatrix

def cosineSimilarity(a, b):
    if type(a) != 'list':
        r = dot(a, b)/(norm1(a)*norm1(b))
    else:
        a = [a]
        b = [b]
        sim = cosine_similarity(a,b)
        r = sim[0][0]
    return r

def effectSize(array, mean):
    new_array = []
    for elem in array:
        new_array.append(float(elem))

    effect = mean/stat.stdev(new_array)
    return effect

def calculateCumulativeProbability(nullDistribution, testStatistic, distribution):
    cumulative = -100
    nullDistribution.sort()

    if distribution == 'empirical':
        ecdf = ECDF(nullDistribution)
        cumulative = ecdf(testStatistic)
    elif distribution == 'normal':
        d = norm(loc = stat.mean(nullDistribution), scale = stat.stdev(nullDistribution))
        cumulative = d.cdf(testStatistic)

    return cumulative

def computeNullHypothesis(wordIndex, nullMatrix, iterations, stereotype1, stereotype2,bothStereotypes):

    print("Number of permutations ", iterations)

    #Assuming both stereotypes have the same length
    setSize = int(len(bothStereotypes)/2)

    toShuffle = list(range(0, len(bothStereotypes)))
    distribution = []

    for iter in range(iterations):
        random.shuffle(toShuffle)
        #calculate mean for each null shuffle
        meanSimilarityGroup1 = 0
        meanSimilarityGroup2 = 0

        for i in range(setSize):
            meanSimilarityGroup1 = meanSimilarityGroup1 + nullMatrix[wordIndex][toShuffle[i]]

        for i in range(setSize):
            meanSimilarityGroup2 = meanSimilarityGroup2 + nullMatrix[wordIndex][toShuffle[i+setSize]]

        meanSimilarityGroup1 = meanSimilarityGroup1/(setSize)
        meanSimilarityGroup2 = meanSimilarityGroup2/(setSize)

        distribution.append(meanSimilarityGroup1 - meanSimilarityGroup2)

    return distribution

def computeNullMatrix(names, bothStereotypes, vectors, word2index):

    concept1NullMatrix = []

    for name in names:

        concept1Embedding = vectors[word2index[name]]
        my_list = []

        for attribute in bothStereotypes:
            nullEmbedding = vectors[word2index[attribute]]
            similarityCompatible = cosineSimilarity(concept1Embedding, nullEmbedding)
            my_list.append(similarityCompatible)
        concept1NullMatrix.append(my_list)

    return concept1NullMatrix

def mean_concept_stereotype(attributesFirstSet, attributesSecondSet,names,
                                                vectors,word2index,vectors_after, word2index_after):

    bothStereotypes = attributesFirstSet + attributesSecondSet

    meanConceptStereotype1 = [0]* len(names)
    meanConceptStereotype2 = [0]* len(names)

    meanConceptStereotype1_after = [0]* len(names)
    meanConceptStereotype2_after = [0]* len(names)

    for i in range(len(names)):

        conceptEmbedding = vectors[word2index[names[i]]]
        conceptEmbedding_after = vectors_after[word2index_after[names[i]]]

        for attribute in attributesFirstSet:
            stereotype2Embedding = vectors[word2index[attribute]]
            similarityCompatible = cosineSimilarity(conceptEmbedding, stereotype2Embedding)
            meanConceptStereotype1[i] = meanConceptStereotype1[i] + similarityCompatible

            stereotype2Embedding_after = vectors_after[word2index_after[attribute]]
            similarityCompatible_after = cosineSimilarity(conceptEmbedding_after, stereotype2Embedding_after)
            meanConceptStereotype1_after[i] = meanConceptStereotype1_after[i] + similarityCompatible_after


        meanConceptStereotype1[i] = meanConceptStereotype1[i]/len(attributesFirstSet)
        meanConceptStereotype1_after[i] = meanConceptStereotype1_after[i]/len(attributesFirstSet)
        for attribute in attributesSecondSet:
            stereotype2Embedding = vectors[word2index[attribute]]
            similarityCompatible = cosineSimilarity(conceptEmbedding, stereotype2Embedding)
            meanConceptStereotype2[i] = meanConceptStereotype2[i] + similarityCompatible

            stereotype2Embedding_after = vectors_after[word2index_after[attribute]]
            similarityCompatible_after = cosineSimilarity(conceptEmbedding_after, stereotype2Embedding_after)
            meanConceptStereotype2_after[i] = meanConceptStereotype2_after[i] + similarityCompatible_after


        meanConceptStereotype2[i] = meanConceptStereotype2[i]/len(attributesSecondSet)
        meanConceptStereotype2_after[i] = meanConceptStereotype2_after[i]/len(attributesSecondSet)

    results = [meanConceptStereotype1, meanConceptStereotype2, meanConceptStereotype1_after, meanConceptStereotype2_after]

    return results

def wefat(names,conceptNullMatrix, conceptNullMatrix_after,vectors,word2index,vectors_after,word2index_after,bothStereotypes, meanConceptStereotype1, meanConceptStereotype2,meanConceptStereotype1_after,meanConceptStereotype2_after,attributesFirstSet,attributesSecondSet, genderList, numIterations):
    distribution = "normal"
    count = 0
    fem_count = 0
    masc_count = 0
    fem_correct = 0
    masc_correct = 0
    p_before = []
    p_after = []
    d_before = []
    d_after = []
    for i in range(len(names)):
        nullDistributionConcept = []
        nullDistributionConcept_after = []

        for j in range(len(bothStereotypes)):
            nullDistributionConcept.append(conceptNullMatrix[i][j])
            nullDistributionConcept_after.append(conceptNullMatrix_after[i][j])

        conceptEmbedding = vectors[word2index[names[i]]]
        conceptEmbedding_after = vectors_after[word2index_after[names[i]]]

        print("WEFAT for word   ", names[i], "gender:   ", genderList[i])
        e1 = effectSize(nullDistributionConcept, meanConceptStereotype1[i] - meanConceptStereotype2[i])
        myMatrix = computeNullHypothesis(i, conceptNullMatrix, numIterations, attributesFirstSet, attributesSecondSet, bothStereotypes)
        cumulativeVal = calculateCumulativeProbability(myMatrix, (meanConceptStereotype2[i] - meanConceptStereotype1[i]), distribution)
        d_before.append(e1)
        p_before.append(cumulativeVal)

        e2 = effectSize(nullDistributionConcept_after, meanConceptStereotype1_after[i] - meanConceptStereotype2_after[i])
        myMatrix_after = computeNullHypothesis(i, conceptNullMatrix_after, numIterations, attributesFirstSet, attributesSecondSet, bothStereotypes)
        cumulativeVal_after = calculateCumulativeProbability(myMatrix_after, (meanConceptStereotype2_after[i] - meanConceptStereotype1_after[i]), distribution)
        print("effect size:     before: ", e1, "    after: ", e2)
        print("p val:       before: ", cumulativeVal, "     after: ", cumulativeVal_after)
        print(i)
        print("\n")
        d_after.append(e2)
        p_after.append(cumulativeVal_after)

        # if genderList[i] == "fem":
        #     fem_count += 1
        #     if e2>e1:
        #         fem_correct += 1
        # elif genderList[i] == "masc":
        #     masc_count += 1
        #     if e2<e1:
        #         masc_correct += 1

    # return fem_correct,fem_count, masc_correct,masc_count
    return p_before, p_after, d_before, d_after
