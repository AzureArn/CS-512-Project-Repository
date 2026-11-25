import csv
import time
from pathlib import Path

import numpy as np

import FromDataFileMLR
import FromFinessFileMLR
import mlr


class R2SortedCSVWriter:
    def __init__(self, file_name):
        self.file_out = open(file_name, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file_out)
        self.header = None
        self.rows = []

    def writerow(self, row):
        if self.header is None:
            self.header = row
        else:
            self.rows.append(row)

    def close(self):
        try:
            if self.header is not None:
                self.writer.writerow(self.header)
            self.rows.sort(key=lambda row: float(row[3]), reverse=True)
            self.writer.writerows(self.rows)
        finally:
            self.file_out.close()


class FitnessAnalyzer:
    def __init__(self, fitnessdata=None, random_state=None):
        self.fitnessdata = fitnessdata or FromFinessFileMLR.FitnessResults()
        self.rng = np.random.default_rng(random_state)

    def getAValidrow(self, numOfFea, eps=0.015):
        if numOfFea <= 0:
            raise ValueError("numOfFea must be greater than zero.")
        if not 0 < eps <= 1:
            raise ValueError("eps must be greater than zero and at most one.")

        return (self.rng.random(numOfFea) < eps).astype(int)

    def Create_A_Population(self, numOfPop, numOfFea):
        if numOfPop <= 0:
            raise ValueError("numOfPop must be greater than zero.")
        population = np.array(
            [self.getAValidrow(numOfFea) for _ in range(numOfPop)],
            dtype=int,
        )
        return population.reshape(numOfPop, numOfFea)

    def createAnOutputFile(self, algorithm=None):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        algorithm_name = algorithm.__class__.__name__ if algorithm is not None else "MLR"
        file_name = Path(__file__).resolve().parent / f"{algorithm_name}_{timestamp}.csv"
        file_writer = R2SortedCSVWriter(file_name)
        file_writer.writerow(
            [
                "Descriptor ID",
                "Fitness",
                "Model",
                "R2",
                "Q2",
                "R2Pred_Validation",
                "R2Pred_Test",
            ]
        )
        return file_writer

    def createANewPopulation(self, numOfPop, numOfFea, OldPopulation, fitness):
        OldPopulation = np.asarray(OldPopulation, dtype=int)
        fitness = np.asarray(fitness, dtype=float)
        if OldPopulation.shape != (numOfPop, numOfFea):
            raise ValueError("OldPopulation shape does not match the requested size.")
        if fitness.shape != (numOfPop,) or not np.all(np.isfinite(fitness)):
            raise ValueError("Fitness must contain one finite value per population row.")
        if numOfPop < 2:
            raise ValueError("At least two population members are needed for selection.")

        parent_indices = np.argsort(fitness)[:2]
        mom = OldPopulation[parent_indices[0]].tolist()
        dad = OldPopulation[parent_indices[1]].tolist()

        crossover_point = int(self.rng.integers(1, numOfFea))
        child1 = mom[:crossover_point] + dad[crossover_point:]
        child2 = dad[:crossover_point] + mom[crossover_point:]
        new_population = [mom, dad, child1, child2][:numOfPop]
        while len(new_population) < numOfPop:
            new_population.append(self.getAValidrow(numOfFea).tolist())
        return np.asarray(new_population, dtype=int)

    def PerformOneThousandIteration(
        self,
        numOdPop,
        numOfFea,
        population,
        fitness,
        model,
        fileW,
        TrainX,
        TrainY,
        ValidateX,
        ValidateY,
        TestX,
        TestY,
        fittingStatus=False,
        R2req_train=0.6,
        Q2req_train=0.5,
        R2req_validate=0.5,
        R2req_test=0.5,
        R2max=1.0,
    ):
        for generation in range(1, 1001):
            if fittingStatus:
                break

            population = self.createANewPopulation(
                numOdPop, numOfFea, population, fitness
            )
            fittingStatus, fitness = self.fitnessdata.validate_model(
                model,
                fileW,
                population,
                TrainX,
                TrainY,
                ValidateX,
                ValidateY,
                TestX,
                TestY,
                R2req_train,
                Q2req_train,
                R2req_validate,
                R2req_test,
                R2max,
            )
            print(f"Generation {generation}")
            if fittingStatus:
                print("Model requirements met; stopping.")


def main():
    model = mlr.MLR()
    filedata = FromDataFileMLR.DataFromFile()
    analyzer = FitnessAnalyzer()
    file_writer = analyzer.createAnOutputFile(model)

    num_of_population = 50
    r2req_train = 0.5
    q2req_train = 0.5
    r2req_validate = 0.5
    r2req_test = 0.5
    r2max = 1.0

    try:
        TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = (
            filedata.getAllOfTheData()
        )
        TrainX, ValidateX, TestX = filedata.rescaleTheData(
            TrainX, ValidateX, TestX
        )
        num_of_features = TrainX.shape[1]
        print("Population size:", num_of_population)
        print("Feature count:", num_of_features)

        population = analyzer.Create_A_Population(
            num_of_population, num_of_features
        )
        fitting_status, fitness = analyzer.fitnessdata.validate_model(
            model,
            file_writer,
            population,
            TrainX,
            TrainY,
            ValidateX,
            ValidateY,
            TestX,
            TestY,
            r2req_train,
            q2req_train,
            r2req_validate,
            r2req_test,
            r2max,
        )
        if fitting_status:
            print("Model requirements met; stopping.")

        analyzer.PerformOneThousandIteration(
            num_of_population,
            num_of_features,
            population,
            fitness,
            model,
            file_writer,
            TrainX,
            TrainY,
            ValidateX,
            ValidateY,
            TestX,
            TestY,
            fitting_status,
            r2req_train,
            q2req_train,
            r2req_validate,
            r2req_test,
            r2max,
        )
    finally:
        file_writer.close()


if __name__ == "__main__":
    main()
