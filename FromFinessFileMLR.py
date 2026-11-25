import numpy as np
from sklearn.model_selection import KFold

import FromDataFileMLR


class FitnessResults:
    def __init__(self):
        self.filedata = FromDataFileMLR.DataFromFile()

    def r2(self, y, y_hat):
        residual_sum = np.sum((y - y_hat) ** 2)
        total_sum = np.sum((y - np.mean(y)) ** 2)
        if total_sum == 0:
            return 1.0 if residual_sum == 0 else 0.0
        return 1 - residual_sum / total_sum

    def r2Pred(self, y_train, y_test, y_hat_test):
        residual_sum = np.sum((y_hat_test - y_test) ** 2)
        total_sum = np.sum((y_test - np.mean(y_train)) ** 2)
        if total_sum == 0:
            return 1.0 if residual_sum == 0 else 0.0
        return 1 - residual_sum / total_sum

    def cv_predict(self, model, set_x, set_y):
        if len(set_y) < 2:
            raise ValueError("Cross-validation requires at least two training rows.")

        predictions = np.empty_like(set_y, dtype=float)
        n_splits = 10 if len(set_y) >= 10 else len(set_y)
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_indices, test_indices in folds.split(set_x):
            model.fit(set_x[train_indices], set_y[train_indices])
            predictions[test_indices] = model.predict(set_x[test_indices]).reshape(-1)
        return predictions

    def calc_fitness(self, y, y_hat):
        return np.sqrt(np.mean((y - y_hat) ** 2))

    def validate_model(
        self,
        model,
        fileW,
        population,
        TrainX,
        TrainY,
        ValidateX,
        ValidateY,
        TestX,
        TestY,
        R2req_train=0.5,
        Q2req_train=0.5,
        R2req_validate=0.5,
        R2req_test=0.5,
        R2max=1.0,
    ):
        population = np.asarray(population)
        TrainY = np.asarray(TrainY).reshape(-1)
        ValidateY = np.asarray(ValidateY).reshape(-1)
        TestY = np.asarray(TestY).reshape(-1)
        if population.ndim != 2 or population.shape[1] != TrainX.shape[1]:
            raise ValueError("Population columns must match the training features.")
        if population.shape[0] == 0:
            raise ValueError("Population must contain at least one candidate.")

        fitness = np.zeros(population.shape[0])
        requirements_met = False

        for population_index, chromosome in enumerate(population):
            selected_features = np.flatnonzero(chromosome)

            train_x = TrainX[:, selected_features]
            validation_x = ValidateX[:, selected_features]
            test_x = TestX[:, selected_features]

            y_hat_cv = self.cv_predict(model, train_x, TrainY)
            model_desc = model.fit(train_x, TrainY)
            y_hat_train = model.predict(train_x).reshape(-1)
            y_hat_validation = model.predict(validation_x).reshape(-1)
            y_hat_test = model.predict(test_x).reshape(-1)

            r2_train = self.r2(TrainY, y_hat_train)
            q2_cv = self.r2(TrainY, y_hat_cv)
            r2_validation = self.r2Pred(TrainY, ValidateY, y_hat_validation)
            r2_test = self.r2Pred(TrainY, TestY, y_hat_test)

            y_fitness = np.concatenate((TrainY, ValidateY))
            y_hat_fitness = np.concatenate((y_hat_cv, y_hat_validation))
            fitness[population_index] = self.calc_fitness(y_fitness, y_hat_fitness)

            fileW.writerow(
                [
                    str(selected_features.tolist()),
                    self.filedata.getTwoDecPoint(fitness[population_index]),
                    model_desc,
                    self.filedata.getTwoDecPoint(r2_train),
                    self.filedata.getTwoDecPoint(q2_cv),
                    self.filedata.getTwoDecPoint(r2_validation),
                    self.filedata.getTwoDecPoint(r2_test),
                ]
            )

            candidate_meets_requirements = all(
                lower <= score < R2max
                for score, lower in (
                    (r2_train, R2req_train),
                    (q2_cv, Q2req_train),
                    (r2_validation, R2req_validate),
                    (r2_test, R2req_test),
                )
            )
            requirements_met = requirements_met or candidate_meets_requirements

        return requirements_met, fitness
