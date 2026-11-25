import time  # provides timing for benchmarks
from numpy import *  # provides complex math and array functions (also numpy's random module)
from sklearn import svm  # provides Support Vector Regression
from sklearn import linear_model
from sklearn import neural_network
from sklearn import cross_decomposition  #provides Partial Least Square Regression
import csv
import math
import sys
import pandas as pd  # used by sort_results at the end of the run

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
        fileOut = open(file_name, 'w', newline='')
        fileW = csv.writer(fileOut)

        fileW.writerow(['Descriptor ID', 'Fitness', 'Model', 'R2', 'Q2', 'R2Pred_Validation', 'R2Pred_Test'])

        # Return the file object and its name too, so main() can close it and sort it at the end
        return fileW, fileOut, file_name
    # -------------------------------------------------------------------------------------------
    def createANewPopulation(self, numOfPop, numOfFea, OldPopulation, fitness):
        # LOWER fitness is better (fitness is an error measure), so parents are the lowest values.
        order = argsort(array(fitness, dtype=float))
        pool = [list(OldPopulation[i]) for i in order[:5]]   # best 5 rows

        MOM = pool[0]
        DAD = pool[1]
        NewPopulation = [MOM, DAD]                            # always keep the two best

        # Single-point crossover children from the pool, about half of the population
        while len(NewPopulation) < numOfPop // 2:
            a = pool[random.randint(0, len(pool))]
            b = pool[random.randint(0, len(pool))]
            crossover_point = random.randint(1, numOfFea - 1)
            for child in (a[:crossover_point] + b[crossover_point:],
                          b[:crossover_point] + a[crossover_point:]):
                if not 5 <= sum(child) <= 15:
                    child = self.getAValidrow(numOfFea)
                if len(NewPopulation) < numOfPop // 2:
                    NewPopulation.append(child)

        # Fill the remaining rows with random valid feature vectors (keeps new features coming in).
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

def sort_results(in_file):
    # Reads the finished results file and saves a sorted copy next to it.
    out_file = in_file.replace(".csv", "_sorted.csv")

    df = pd.read_csv(in_file)

    # Highest training R2 first; NaN rows go to the bottom.
    # To put the best model (lowest fitness) on top instead, use:
    #   df = df.sort_values("Fitness", ascending=True, na_position="last")
    df = df.sort_values("R2", ascending=False, na_position="last")

    df.to_csv(out_file, index=False)

    # Show the top 10 with all columns visible
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(df.head(10))
    print(f"\nSorted {len(df)} rows -> {out_file}")


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
    fileW, fileOut, file_name = analyzer.createAnOutputFile()

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

    # Optional: uncomment to make runs repeatable, e.g. when comparing two versions of the GA
    # random.seed(1)

    # getAllOfTheData is in FromDataFileMLR file. The following places the data
    # (training data, validation data, and test data) into associated matrices
    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = filedata.getAllOfTheData() # gets descriptor values and target values
    # Note that rescalling is already done in your cleaning section and you do not need to do it again
    TrainX, ValidateX, TestX = filedata.rescaleTheData(TrainX, ValidateX, TestX)

    # Quick sanity checks on the data (safe to delete once you trust the data)
    print(TrainX.shape, ValidateX.shape, TestX.shape)
    print("NaN in TrainX:", isnan(TrainX).any())
    print("Y mean:", TrainY.mean(), ValidateY.mean(), TestY.mean())
    print("Y std:", TrainY.std(), ValidateY.std(), TestY.std())
    print("max abs X:", abs(TrainX).max(), abs(ValidateX).max(), abs(TestX).max())

    numOfFea = TrainX.shape[1]
    print("Population size:", numOfPop)
    print("Feature count:", numOfFea)

    fittingStatus = unfit
    population = analyzer.Create_A_Population(numOfPop, numOfFea)
    fittingStatus, fitness = fitnessdata.validate_model(model, fileW, population,
                                            TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

    try:
        analyzer.PerformOneThousandIteration(numOfPop, numOfFea, population, fitness, model, fileW,
                                   TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
    finally:
        fileOut.close()           # flush everything to disk first
        sort_results(file_name)   # then write the sorted copy


# main routine ends in here
# ------------------------------------------------------------------------------
main()
# ------------------------------------------------------------------------------
