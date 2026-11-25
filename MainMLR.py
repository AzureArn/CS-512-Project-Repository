import time  # provides timing for benchmarks
from numpy import *  # provides complex math and array functions
from sklearn import svm  # provides Support Vector Regression
from sklearn import linear_model
from sklearn import neural_network
from sklearn import cross_decomposition  #provides Partial Least Square Regression
import csv
import math
import sys

# Local files created by me
import mlr
import FromDataFileMLR
import FromFinessFileMLR

class FitnessAnalyzer:
    def __init__(self):
        self.fitnessdata = FromFinessFileMLR.FitnessResults()
    # ------------------------------------------------------------------------------
    def getAValidrow(self, numOfFea, eps=None):
        # In the parameters, numOfFea refers to the number of columns in the data. 
        # This number is total number of your columns. In this routune, you need to
        # select a random number between 0 and 1: {r = random.uniform(0, 1)}.
        # If the value is less that or equal 0.015 (1.5% of the features) you can
        # set that index to 1; otherwise, the index is set to zero.
        # After going through the entire features (bits), you can add all the numbers
        # together to see if at least 5 features and at most 15 features are set to 1.
        # If that is the case you can return the vector; otherwise, you need to do it again.
        
        # Build complete feature vectors and keep only those containing 5 to 15 features.
        # By default, aim for 10 selected features across numOfFea columns.
        if eps is None:
            eps = 10.0 / numOfFea

        while True:
            V = [1 if random.uniform(0, 1) < eps else 0 for i in range(numOfFea)]
            selected = sum(V)
            if 5 <= selected <= 15:
                return V
    # ------------------------------------------------------------------------------
    def Create_A_Population(self, numOfPop, numOfFea):
        # this routine creates the first poprlation. We want to have 50 rows
        # (50 population) and number of features is the number of your columns.
        # Since this the first population, all 50 rows should be randomly selected. 
        # Therefore, for each row, you need to call the getAValidrow function 
        # and move the vector into the row. Once the first population is created
        # you can return it to Callee
        population = []
        for i in range(numOfPop): #how he said to write it on the board in class
            v = self.getAValidrow(numOfFea)
            population.append(v)

        return array(population, dtype=int)
    # ------------------------------------------------------------------------------
    # The following creates an output file. Every time a model is created the
    # descriptors of the model, the name of the model (ex: "MLR" for multiple
    # linear regression of "SVM" support vector machine), the R^2 of training, Q^2
    # of training,R^2 of validation, and R^2 of test is placed in the output file

    def createAnOutputFile(self):
        file_name = None
        algorithm = None

        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        if ((file_name == None) and (algorithm != None)):
            file_name = "{}_{}_gen{}_{}.csv".format(algorithm.__class__.__name__,
                                                    algorithm.model.__class__.__name__, algorithm.gen_max, timestamp)
        elif file_name == None:
            file_name = "{}.csv".format(timestamp)
        fileOut = open(file_name, 'w')
        fileW = csv.writer(fileOut)

        fileW.writerow(['Descriptor ID', 'Fitness', 'Model', 'R2', 'Q2', 'R2Pred_Validation', 'R2Pred_Test'])

        return fileW
    # -------------------------------------------------------------------------------------------
    def createANewPopulation(self, numOfPop, numOfFea, OldPopulation, fitness):
        # Convert fitness values to a Python list and find the two best rows.
        fitness = list(fitness)

        mom_index = fitness.index(max(fitness))
        fitness_without_mom = fitness.copy()
        fitness_without_mom[mom_index] = -math.inf
        dad_index = fitness_without_mom.index(max(fitness_without_mom))

        MOM = list(OldPopulation[mom_index])
        DAD = list(OldPopulation[dad_index])

        # Create two children using a randomly selected crossover point.
        crossover_point = random.randint(1, numOfFea - 1)
        child1 = MOM[:crossover_point] + DAD[crossover_point:]
        child2 = DAD[:crossover_point] + MOM[crossover_point:]

        if not 5 <= sum(child1) <= 15:
            child1 = self.getAValidrow(numOfFea)
        if not 5 <= sum(child2) <= 15:
            child2 = self.getAValidrow(numOfFea)

        NewPopulation = [MOM, DAD, child1, child2]

        # Fill the remaining rows with random valid feature vectors.
        while len(NewPopulation) < numOfPop:
            NewPopulation.append(self.getAValidrow(numOfFea))

        return array(NewPopulation, dtype=int)
        
    # -------------------------------------------------------------------------------------------
    def PerformOneThousandIteration(self, numOdPop, numOfFea, population, fitness, model, fileW,
                                   TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):
        NumOfGenerations = 0
        OldPopulation = population

        while NumOfGenerations < 1000:
            population = self.createANewPopulation(numOdPop, numOfFea, OldPopulation, fitness)
            fittingStatus, fitness = self.fitnessdata.validate_model(model, fileW, population,
                                            TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
            OldPopulation = population
            NumOfGenerations += 1
            print(NumOfGenerations)
        return
    # --------------------------------------------------------------------------------------------

def main():
    # create an object of Multiple Linear Regression model.
    # The class is located in mlr file
    model = mlr.MLR()
    #model = svm.LinearSVR()
    #model = cross_decomposition.PLSRegression()

    filedata = FromDataFileMLR.DataFromFile()           # constructor for FromDataFileMLR class
    fitnessdata = FromFinessFileMLR.FitnessResults()    # constructor for FromFitnessFileMLR object
    analyzer = FitnessAnalyzer()                        # constructor for MainMLR class

    # create an output file. Name the object to be FileW
    fileW = analyzer.createAnOutputFile()

    # Number of population rows; the feature count comes from the loaded training data.
    numOfPop = 50

    # we continue exhancing the model; however if after 1000 iteration no
    # enhancement is done, we can quit
    unfit = 1000

    # Final model requirements: The following is used to evaluate each model. The minimum
    # values for R^2 of training should be 0.6, R^2 of Validation should be 0.5 and R^2 of
    # test should be 0.5
    R2req_train = .6
    R2req_validate = .5
    R2req_test = .5

    # getAllOfTheData is in FromDataFileMLR file. The following places the data
    # (training data, validation data, and test data) into associated matrices
    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = filedata.getAllOfTheData() # gets descriptor values and target values
    # Note that rescalling is already done in your cleaning section and you do not need to do it again
    TrainX, ValidateX, TestX = filedata.rescaleTheData(TrainX, ValidateX, TestX)

    numOfFea = TrainX.shape[1]
    print("Population size:", numOfPop)
    print("Feature count:", numOfFea)

    fittingStatus = unfit
    population = analyzer.Create_A_Population(numOfPop, numOfFea)
    fittingStatus, fitness = fitnessdata.validate_model(model, fileW, population,
                                            TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

    analyzer.PerformOneThousandIteration(numOfPop, numOfFea, population, fitness, model, fileW,
                               TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)


# main routine ends in here
# ------------------------------------------------------------------------------
main()
# ------------------------------------------------------------------------------


