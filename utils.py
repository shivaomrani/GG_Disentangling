from sklearn.metrics.pairwise import cosine_similarity
import random as random
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
import statistics as stat
from numpy import dot
from numpy.linalg import norm as norm1
from statistics import mean

class operations(object):

    def __init__(self, iterations, vectors, word2index, distribution,stereotype1,stereotype2,concept1,concept2=[]):
        self.concept1 = concept1
        self.concept2 = concept2
        self.stereotype1 = stereotype1
        self.stereotype2 = stereotype2
        self.iterations = iterations
        self.vectors = vectors
        self.word2index = word2index
        self.distribution = distribution

    def get_embedding(self, word):
        emb = self.vectors[self.word2index[word]]
        return emb

    def perform_weat(self):
        results = self.getPValueAndEffect()
        print("p-value: ", results[0], "  ---  effectSize: ", results[1])
        return results

    def perform_sc_weat(self):
        difference_of_means_list = self.compute_mean_difference_list(self.concept1)
        conceptNullMatrix = self.compute_null_matrix(self.stereotype1+self.stereotype2)
        d_list, p_list = self.get_pval_effect_sc_weat(conceptNullMatrix, difference_of_means_list)
        return d_list, p_list

    def get_pval_effect_sc_weat(self, conceptNullMatrix, difference_of_means_list):
        d_list = []
        p_list = []
        bothStereotypes = self.stereotype1+self.stereotype2
        for i in range(len(self.concept1)):
            nullDistributionConcept = []

            for j in range(len(bothStereotypes)):
                nullDistributionConcept.append(conceptNullMatrix[j][i])


            e1 = self.effectSize(nullDistributionConcept, difference_of_means_list[i])
            myMatrix = self.compute_null_sc_weat(i, conceptNullMatrix)
            cumulativeVal = self.calculateCumulativeProbability(myMatrix, difference_of_means_list[i])
            d_list.append(e1)
            p_list.append(1-cumulativeVal)

        return d_list, p_list

    def getPValueAndEffect(self):
        results = []
        testStatistic = self.getTestStatistic()
        nullDist = self.nullDistribution()
        entireDistribution = self.getEntireDistribution()
        results.append(1-self.calculateCumulativeProbability(nullDist, testStatistic))
        results.append(self.effectSize(entireDistribution, testStatistic))
        results.append(stat.stdev(nullDist))
        return results

    def nullDistribution(self):

        # permute concepts and for each permutation calculate getTestStatistic and save it in your distribution
        bothConcepts = self.concept1 + self.concept2
        print("Generating null distribution...")

        stereotype1NullMatrix = self.compute_null_matrix(self.stereotype1)
        stereotype2NullMatrix = self.compute_null_matrix(self.stereotype2)

        #Assuming both concepts have the same length
        setSize = int(len(bothConcepts)/2)
        print("Number of permutations ", self.iterations)
        toShuffle = list(range(0, len(bothConcepts)))
        distribution = []

        for iter in range(self.iterations):
            random.shuffle(toShuffle)
            #calculate mean for each null shuffle
            meanSimilaritycon1str1 = self.calculate_mean(self.stereotype1, self.concept1, stereotype1NullMatrix, toShuffle, setSize)
            meanSimilaritycon1str2 = self.calculate_mean(self.stereotype2, self.concept1, stereotype2NullMatrix, toShuffle, setSize)
            meanSimilaritycon2str1 = self.calculate_mean(self.stereotype1, self.concept2, stereotype1NullMatrix, toShuffle, setSize)
            meanSimilaritycon2str2 = self.calculate_mean(self.stereotype2, self.concept2, stereotype2NullMatrix, toShuffle, setSize)

            #come back here later
            distribution.append((meanSimilaritycon1str1 - meanSimilaritycon1str2) - meanSimilaritycon2str1 + meanSimilaritycon2str2)

        return distribution

    def calculate_mean(self, stereotype, concept, stereotypeNullMatrix, toShuffle, setSize):
        meanSimilarityconstr_list = []
        for i in range(len(stereotype)):
            for j in range(setSize):
                if concept == self.concept2 or concept == self.stereotype2:
                    meanSimilarityconstr_list.append(stereotypeNullMatrix[i][toShuffle[j+setSize]])
                elif concept == self.concept1:
                    meanSimilarityconstr_list.append(stereotypeNullMatrix[i][toShuffle[j]])

        meanSimilarityconstr = mean(meanSimilarityconstr_list)
        return meanSimilarityconstr


    def compute_null_matrix(self, stereotype):
        stereotypeNullMatrix = []
        bothConcepts = self.concept1 + self.concept2

        for attribute in stereotype:
            similarity_list = []
            stereotypeEmbedding = self.get_embedding(attribute)

            for word in bothConcepts:
                nullEmbedding = self.get_embedding(word)
                similarity = self.cosineSimilarity(nullEmbedding, stereotypeEmbedding)
                similarity_list.append(similarity)
            stereotypeNullMatrix.append(similarity_list)
        return stereotypeNullMatrix

    def calculateCumulativeProbability(self,nullDistribution, testStatistic):
        cumulative = -100
        nullDistribution.sort()

        if self.distribution == 'empirical':
            ecdf = ECDF(nullDistribution)
            cumulative = ecdf(testStatistic)
        elif self.distribution == 'normal':
            d = norm(loc = stat.mean(nullDistribution), scale = stat.stdev(nullDistribution))
            cumulative = d.cdf(testStatistic)

        return cumulative

    def effectSize(self,array, mean):
        new_array = [float(elem) for elem in array]
        effect = mean/stat.stdev(new_array)
        return effect

    def getTestStatistic(self):

        differenceOfMeansConcept1_list = self.compute_mean_difference_list(self.concept1)
        # effect size computations mean S(x,A,B)
        differenceOfMeansConcept1 = mean(differenceOfMeansConcept1_list)

        differenceOfMeansConcept2_list = self.compute_mean_difference_list(self.concept2)
        differenceOfMeansConcept2 = mean(differenceOfMeansConcept2_list)

        differenceOfMeans = differenceOfMeansConcept1 - differenceOfMeansConcept2
        #used for effect size computations before dividing by standard deviation
        print("The difference of means is ", differenceOfMeans)
        return differenceOfMeans

    def compute_mean_difference_list(self, concept):
        differenceOfMeans_list = []
        # concept computations
        for word in concept:
            concept_embedding = self.get_embedding(word)
            meanConceptStereotype1 = self.compute_mean_concept_stereotype_similarity(self.stereotype1, concept_embedding)
            meanConceptStereotype2 = self.compute_mean_concept_stereotype_similarity(self.stereotype2, concept_embedding)
            differenceOfMeans_list.append(meanConceptStereotype1 - meanConceptStereotype2)

        return differenceOfMeans_list

    def compute_mean_concept_stereotype_similarity(self, stereotype, emb):
        meanConceptStereotype_list = []
        for attribute in stereotype:
            stereotype_embedding = self.get_embedding(attribute)
            similarity = self.cosineSimilarity(emb, stereotype_embedding)
            meanConceptStereotype_list.append(similarity)

        meanConceptStereotype = mean(meanConceptStereotype_list)
        return meanConceptStereotype

    def getEntireDistribution(self):

        bothConcepts = self.concept1 + self.concept2
        distribution = []
        print("Getting the entire distribution")

        for word in bothConcepts:
            emb = self.get_embedding(word)
            similarityToStereotype1 = self.compute_mean_concept_stereotype_similarity(self.stereotype1, emb)
            similarityToStereotype2 = self.compute_mean_concept_stereotype_similarity(self.stereotype2, emb)
            distribution.append(similarityToStereotype1 - similarityToStereotype2)

        return distribution


    def cosineSimilarity(self,a, b):
        if type(a) != 'list':
            r = dot(a, b)/(norm1(a)*norm1(b))
        else:
            a = [a]
            b = [b]
            sim = cosine_similarity(a,b)
            r = sim[0][0]
        return r

    def compute_null_sc_weat(self, index, nullMatrix):

        bothStereotypes = self.stereotype1+self.stereotype2

        # Assuming both stereotypes have the same length
        setSize = int(len(bothStereotypes) / 2)
        toShuffle = list(range(0, len(bothStereotypes)))
        distribution = []

        for iter in range(self.iterations):
            random.shuffle(toShuffle)
            # calculate mean for each null shuffle
            meanSimilarityGroup1_list = []
            meanSimilarityGroup2_list = []

            for i in range(setSize):
                meanSimilarityGroup1_list.append(nullMatrix[toShuffle[i]][index])
                meanSimilarityGroup2_list.append(nullMatrix[toShuffle[i + setSize]][index])

            meanSimilarityGroup1 = mean(meanSimilarityGroup1_list)
            meanSimilarityGroup2 = mean(meanSimilarityGroup2_list)

            distribution.append(meanSimilarityGroup1 - meanSimilarityGroup2)

        return distribution