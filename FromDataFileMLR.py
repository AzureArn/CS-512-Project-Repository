import csv
from pathlib import Path

import numpy as np


class DataFromFile:
    def getTwoDecPoint(self, x):
        return float(f"{x:.2f}")

    def placeDataIntoArray(self, fileName):
        input_file = Path(__file__).resolve().parent / fileName
        with input_file.open(mode="r", newline="", encoding="utf-8") as csvfile:
            data = np.asarray(list(csv.reader(csvfile)), dtype=np.float64)
        if data.size == 0:
            raise ValueError(f"Data file is empty: {input_file}")
        return data

    def getAllOfTheData(self):
        TrainX = self.placeDataIntoArray("TrainX.csv")
        TrainY = self.placeDataIntoArray("TrainY.csv").reshape(-1)
        ValidateX = self.placeDataIntoArray("ValidX.csv")
        ValidateY = self.placeDataIntoArray("ValidY.csv").reshape(-1)
        TestX = self.placeDataIntoArray("TestX.csv")
        TestY = self.placeDataIntoArray("TestY.csv").reshape(-1)

        datasets = (
            ("training", TrainX, TrainY),
            ("validation", ValidateX, ValidateY),
            ("test", TestX, TestY),
        )
        for name, features, targets in datasets:
            if features.ndim != 2 or features.shape[1] != TrainX.shape[1]:
                raise ValueError(f"{name} features must have {TrainX.shape[1]} columns.")
            if features.shape[0] != targets.shape[0]:
                raise ValueError(f"{name} feature and target row counts do not match.")

        return TrainX, TrainY, ValidateX, ValidateY, TestX, TestY

    def rescaleTheData(self, TrainX, ValidateX, TestX):
        if TrainX.shape[0] < 2:
            raise ValueError("At least two training rows are required for rescaling.")

        train_mean = TrainX.mean(axis=0)
        train_std = TrainX.std(axis=0, ddof=1)
        train_std[train_std == 0] = 1.0

        TrainX = (TrainX - train_mean) / train_std
        ValidateX = (ValidateX - train_mean) / train_std
        TestX = (TestX - train_mean) / train_std
        return TrainX, ValidateX, TestX
