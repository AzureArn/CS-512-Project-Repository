#### Imported Libraries ####
import numpy as np                                  ##from project file
import pandas as pd                                 ##from project file
from sklearn.impute import SimpleImputer            ##from project file
from sklearn.preprocessing import StandardScaler    ##from project file

import sys
import csv
from itertools import zip_longest
from math import isfinite
from pathlib import Path
from openpyxl import load_workbook



#### Creating Matix Object ####
class myMatrix:

        # ============================================================
        # 1. LOAD THE DATA
        # ============================================================
        def GetMatrix(self, file_name):
            input_file = Path(__file__).with_name(file_name)
            workbook = load_workbook(input_file, read_only=True, data_only=True)
            try:
                sheet = workbook.active
                return tuple(
                    tuple("" if value is None else str(value).strip() for value in row)
                    for row in sheet.iter_rows(values_only=True)
                )
            finally:
                workbook.close()

        # ============================================================
        # 2. CONVERT NON-NUMERIC VALUES TO NaN
        # ============================================================
        # Convert numeric fields to numbers, coercing invalid values to NaN.
        # Keep NAME as text so it remains an identifier, not a feature.

        def EditPart2(self, original_rows):

                header = original_rows[0]                                       ## Used to keep Header Row
                data = pd.DataFrame(original_rows[1:], columns=header)
                name_column = next(
                    (column for column in data.columns if str(column).strip().upper() == "NAME"),
                    None,
                )
                name_index = header.index(name_column) if name_column is not None else None
                numeric_columns = [
                    column for column in data.columns if column != name_column
                ]
                data[numeric_columns] = data[numeric_columns].apply(    ## Code from Lab Doc
                    pd.to_numeric,                                              ##
                    errors="coerce",                                            ##
                )                                                               ##
                print("\nNumber of NaN values in each column:")                 ##
                print(data.isna().sum())                                        ##

                def clean_value(value):
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        return value

                    if not isfinite(number):
                        return np.nan
                    return int(number) if number.is_integer() else number

                original_rows = tuple(
                    tuple(
                        value if index == name_index else clean_value(value)
                        for index, value in enumerate(row)
                    )
                    for row in data.itertuples(index=False, name=None)
                )

                def is_valid(values):
                    try:
                        value = float(values[0])
                        valid = isfinite(value) and value != 0.0                                            ## Check target value and reject zeros and junk
                    except (IndexError, ValueError):
                        valid = False                                                                       ## flag if data is empty or non numeric

                    return valid

                valid_rows = tuple(filter(is_valid, original_rows))
                return (header,) + valid_rows

        # ============================================================
        # 3. REMOVE ROWS WHERE Y (IC50) IS NaN
        # ============================================================
        # The first column is our target, Y.
        # Remove rows where IC50 is missing.

        def EditPart3(self, part2):
            if not part2:
                return ()

            header = part2[0]
            data = pd.DataFrame(part2[1:], columns=header)
            target = pd.to_numeric(data.iloc[:, 0], errors="coerce")
            valid_target = target.notna() & (target > 0)  # log needs positive values
            data = data.loc[valid_target].copy()

            # IC50 (nM) -> pIC50
            data.iloc[:, 0] = (9 - np.log10(target[valid_target])).values

            # Remove compounds that have no descriptors at all (every X is "na")
            x_numeric = data.iloc[:, 3:].apply(pd.to_numeric, errors="coerce")  # columns: IC50, No., NAME, then X
            empty_rows = x_numeric.isna().mean(axis=1) > 0.9
            print("Rows with no descriptors removed:", empty_rows.sum())
            data = data.loc[~empty_rows]

            valid_rows = tuple(data.itertuples(index=False, name=None))
            return (header,) + valid_rows


        # ============================================================
        # 4. REMOVE X COLUMNS WITH MORE THAN 50% NaN
        # ============================================================
        def EditPart4(self, part3):
            if not part3:
                return ()

            header = part3[0]
            data = pd.DataFrame(part3[1:], columns=header)

            nan_fraction = data.isna().mean()

            print("Shape before drop:", data.shape)
            print(nan_fraction.sort_values(ascending=False).head(10))

            #check how many columns would be dropped at each threshold
            for t in (0.3, 0.5, 0.6, 0.7, 0.9):
                print(f"columns with > {t:.0%} NaN:", (nan_fraction > t).sum())

            #60% is selected as the lowest precent that doesn't drop everything
            columns_to_remove = nan_fraction[nan_fraction> 0.6].index
            print("\nX columns removed (>50% NaN):", list(columns_to_remove))

            name_column = next(
                (column for column in data.columns if str(column).strip().upper() == "NAME"),
                None,
            )
            if name_column is not None:
                columns_to_remove = columns_to_remove.drop(
                    name_column,
                    errors="ignore",
                )
            data = data.drop(columns = columns_to_remove)

            return (tuple(data.columns),) + tuple(data.itertuples(index=False, name=None))


        # ============================================================
        # 5. REMOVE DUPLICATE ROWS
        # ============================================================

        # Remove rows that are completely identical.
        #
        # This prevents the same observation from being counted
        # multiple times during subsequent calculations.
        def EditPart5(self, part4):
            if not part4:
                return ()
            header = part4[0]
            data = pd.DataFrame(part4[1:], columns=header)

            id_col = data.columns[1]
            print("Dropping ID column:", id_col)
            data = data.drop(columns=[id_col])

            number_before = len(data)

            data = data.drop_duplicates()

            number_after = len(data)

            print("\nDuplicate rows removed:", number_before - number_after)
            print("Shape:", data.shape)

            return (tuple(data.columns),) + tuple(data.itertuples(index=False, name=None))

        # ============================================================
        # 6. REMOVE CONSTANT AND NEAR-CONSTANT X COLUMNS
        # ============================================================
        def EditPart6(self, part5):
            if not part5:
                return ()
            header = part5[0]
            data = pd.DataFrame(part5[1:], columns=header)

            # Separate Y and X.
            Y = data.iloc[:, 0].copy()
            name_column = next(
                (column for column in data.columns if str(column).strip().upper() == "NAME"),
                None,
            )
            name = data[name_column].copy() if name_column is not None else None
            x_columns = [
                column for column in data.columns[1:] if column != name_column
            ]
            X = data[x_columns].copy()


            # ------------------------------------------------------------
            # 6a. Remove CONSTANT columns
            # ------------------------------------------------------------

            # A constant column has only one unique value.
            #
            # dropna=True means that NaN values are ignored when
            # determining the number of unique values.

            constant_columns = [ ]

            for column in X.columns:
                number_unique = X[column].nunique(dropna=True)
                if number_unique <= 1:
                    constant_columns.append(column)


            print("\nConstant X columns:")
            print(constant_columns)

            # Remove constant columns.
            X = X.drop(columns=constant_columns)


            # ------------------------------------------------------------
            # 6b. Remove NEAR-CONSTANT columns
            # ------------------------------------------------------------

            # Define the threshold.
            #
            # Here, if 90% or more of the non-NaN values are identical,
            # we consider the variable near-constant.

            near_constant_threshold = 0.525 # 52.5% near constant was the highest threshold that had 0 constant column

            near_constant_columns = [ ]

            for column in X.columns:
                # Calculate frequency of each value.
                frequencies = X[column].value_counts(normalize=True, dropna=True)

                # Check the most common value.
                if len(frequencies) > 0:
                    highest_frequency = frequencies.iloc[0]
                    if highest_frequency >= near_constant_threshold:
                        near_constant_columns.append(column)


            print("\nNear-constant X columns:")
            print(near_constant_columns)

            # Remove near-constant columns.
            X = X.drop(columns=near_constant_columns)

            print("\nNumber of X variables remaining:", X.shape[1])

            parts = [Y]
            if name is not None:
                parts.append(name)
            parts.append(X)
            data = pd.concat(parts, axis=1)
            return (tuple(data.columns),) + tuple(data.itertuples(index=False, name=None))

        # ============================================================
        # 7. DETECT OUTLIERS IN EACH X AND MAKE THEM NaN
        # ============================================================

        # We use the IQR method.
        #
        # Q1  = 25th percentile
        # Q3  = 75th percentile
        # IQR = Q3 - Q1
        #
        # Values below:
        #
        #       Q1 - 1.5 * IQR
        #
        # or above:
        #
        #       Q3 + 1.5 * IQR
        #
        # are considered outliers.
        #
        # IMPORTANT:
        # We only do this for X.
        # We do NOT automatically remove IC50 outliers.
        def EditPart7(self, part6):
            if not part6:
                return ()
            header = part6[0]
            data = pd.DataFrame(part6[1:], columns=header)

            zero_iqr_columns = 0
            wiped_nonzero = 0

            Y = data.iloc[:, 0].copy()
            name_column = next(
                (column for column in data.columns if str(column).strip().upper() == "NAME"),
                None,
            )
            name = data[name_column].copy() if name_column is not None else None
            x_columns = [
                column for column in data.columns[1:] if column != name_column
            ]
            X = data[x_columns].copy()

            total_outliers = 0      #counter for number of outliers

            for column in X.columns:

                # Calculate first quartile.
                Q1 = X[column].quantile(0.25)

                # Calculate third quartile.
                Q3 = X[column].quantile(0.75)

                # Calculate interquartile range.
                IQR = Q3 - Q1

                # Calculate lower and upper limits.
                lower_limit = Q1 - 1.5 * IQR
                upper_limit = Q3 + 1.5 * IQR

                # Identify outliers.
                outliers = ((X[column] < lower_limit) | (X[column] > upper_limit))

                if IQR == 0:
                    zero_iqr_columns += 1
                    wiped_nonzero += outliers.sum()

                total_outliers += outliers.sum()

                # Replace outliers with NaN.
                X.loc[outliers, column] = np.nan


            print("\nOutliers in X have been converted to NaN. ", total_outliers)

            parts = [Y]
            if name is not None:
                parts.append(name)
            parts.append(X)
            data = pd.concat(parts, axis=1)

            print("Columns with IQR == 0:", zero_iqr_columns, "of", X.shape[1])
            print("Values wiped in those columns:", wiped_nonzero, "of", total_outliers, "total outliers")

            return (tuple(data.columns),) + tuple(data.itertuples(index=False, name=None))


        # ============================================================
        # 8. IMPUTE REMAINING NaN VALUES IN X
        # ============================================================
        #Steps 8 and 9 should ultimately be fitted only on your training data, not the entire dataset. Otherwise, the median imputation and scaling parameters contain information from your future test set.
        # At this point, X can contain NaN values from:
        #
        # 1. Original missing values
        # 2. Non-numeric values
        # 3. Outliers that we just converted to NaN
        #
        # We replace missing values using the MEDIAN of each X.
        #
        # Median is generally less sensitive to extreme values
        # than the mean.

        def EditPart8(self, part7):
            if not part7:
                return ()
            header = part7[0]
            data = pd.DataFrame(part7[1:], columns=header)

            Y = data.iloc[:, 0].copy()
            name_column = next(
                (column for column in data.columns if str(column).strip().upper() == "NAME"),
                None,
            )
            name = data[name_column].copy() if name_column is not None else None
            x_columns = [
                column for column in data.columns[1:] if column != name_column
            ]
            X = data[x_columns].copy()

            print("Fraction of X cells that are NaN before imputation:", X.isna().mean().mean())
            print("Columns with more than 30% NaN:", (X.isna().mean() > 0.3).sum(), "of", X.shape[1])

            imputer = SimpleImputer(strategy="median")

            X_imputed = imputer.fit_transform(X)

            # Convert back to a DataFrame so that we keep
            # the original X column names.
            X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

            print("\nRemaining NaN values in X:", X.isna().sum().sum())

            constant_after = [c for c in X.columns if X[c].nunique() <= 1]
            print("Constant columns after imputation:", len(constant_after), "of", X.shape[1])

            parts = [Y]
            if name is not None:
                parts.append(name)
            parts.append(X)
            data = pd.concat(parts, axis=1)
            return (tuple(data.columns),) + tuple(data.itertuples(index=False, name=None))


        # ============================================================
        # 9. SCALE THE X VARIABLES
        # ============================================================
        #Steps 8 and 9 should ultimately be fitted only on your training data, not the entire dataset. Otherwise, the median imputation and scaling parameters contain information from your future test set.
        # StandardScaler performs:
        #
        #       X_scaled = (X - mean) / standard deviation
        #
        # Therefore, each X will have approximately:
        #
        #       mean = 0
        #       standard deviation = 1

        def EditPart9(self, part8):
            if not part8:
                return ()
            header = part8[0]
            data = pd.DataFrame(part8[1:], columns=header)
            Y = data.iloc[:, 0].copy()
            name_column = next(
                (column for column in data.columns if str(column).strip().upper() == "NAME"),
                None,
            )
            name = data[name_column].copy() if name_column is not None else None
            x_columns = [
                column for column in data.columns[1:] if column != name_column
            ]
            X = data[x_columns].copy()

            scaler = StandardScaler()

            X_scaled_array = scaler.fit_transform(X)

            # Convert back to a DataFrame.
            X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)

            if name is not None:
                X_scaled.insert(0, name_column, name)

            print("\nX variables have been standardized.")

            return self.EditPart10(Y, X_scaled)




    # ============================================================
    # 10. COMBINE Y AND CLEANED/SCALED X
    # ============================================================
    # Put the original Y (IC50) back together with
    # the cleaned and scaled X variables.
        def EditPart10(self, Y, X_scaled):
            cleaned_data = pd.concat([Y, X_scaled], axis=1)
            return (tuple(cleaned_data.columns),) + tuple(
                cleaned_data.itertuples(index=False, name=None)
            )


# ============================================================
# CHECK THE FINAL DATASET
# ============================================================
        def Editfinal(self, part10):
            if not part10:
                return ()
            header = part10[0]
            data = pd.DataFrame(part10[1:], columns=header)
            Y = data.iloc[:, 0].copy()
            X = data.iloc[:, 1:].copy()

            cleaned_data = pd.concat([Y, X], axis=1)

            print("\n========================================")
            print("FINAL DATASET")
            print("========================================")

            print("Number of rows:", cleaned_data.shape[0])
            print("Number of columns:", cleaned_data.shape[1])

            print("\nFirst five rows:")
            print(cleaned_data.head())

            print("\nRemaining NaN values:")
            print(cleaned_data.isna().sum().sum())

            return (tuple(cleaned_data.columns),) + tuple(
                cleaned_data.itertuples(index=False, name=None)
            )

# ============================================================
# SAVE THE CLEANED DATA
# ============================================================
def PrintMatrix(self, title, matrix, max_rows=5, max_columns=8):
        row_count = len(matrix)
        column_count = max(map(len, matrix), default=0)
        preview = tuple(map(
            lambda entry: f"{entry[0]:>4} | " + " | ".join(map(str, entry[1][:max_columns])),
            enumerate(matrix[:max_rows], start=1),
        ))
        print(f"{title} ({row_count} rows x {column_count} columns)")
        print(f"Preview: up to {max_rows} rows and {max_columns} columns")
        print("\n".join(preview))


def SaveSplitFiles(final_matrix):
    if len(final_matrix) < 2:
        raise ValueError("Cannot create train, validation, and test files without data rows.")

    header = final_matrix[0]
    name_index = next(
        (index for index, column in enumerate(header)
         if str(column).strip().upper() == "NAME"),
        None,
    )
    x_indices = [
        index for index in range(1, len(header)) if index != name_index
    ]

    # Split names match the files MainMLR.py reads:
    # Train-Data.csv, Train-pIC50.csv, Validation-Data.csv, Validation-pIC50.csv,
    # Test-Data.csv, Test-pIC50.csv
    splits = {
        "Train": ([], []),
        "Validation": ([], []),
        "Test": ([], []),
    }

    # IC50 in nM is inversely related to pIC50, so descending IC50 sorts ascending pIC50.
    sorted_rows = sorted(final_matrix[1:], key=lambda row: float(row[0]), reverse=True)
    split_by_position = ("Train", "Train", "Validation", "Test")
    for row_index, row in enumerate(sorted_rows):
        split_name = split_by_position[row_index % len(split_by_position)]
        x_rows, y_rows = splits[split_name]
        x_rows.append(tuple(row[index] for index in x_indices))
        y_rows.append((row[0],))

    for split_name, (x_rows, y_rows) in splits.items():
        for suffix, rows in (("Data", x_rows), ("pIC50", y_rows)):
            output_file = Path(__file__).with_name(f"{split_name}-{suffix}.csv")
            with output_file.open("w", newline="", encoding="utf-8") as csv_file:
                csv.writer(csv_file).writerows(rows)
            print(f"CSV file saved to: {output_file.name}")



### Save CSV Files ###
def SaveMatrices(file_name):
    matrix = myMatrix()                                                 ## Create the Matrix class
    Part2 = myMatrix.EditPart2(matrix, matrix.GetMatrix(file_name))     ## Call Part2 to remove non numerics from data
    Part3 = myMatrix.EditPart3(matrix, Part2)                           ## Call Part3 to remove rows where Y (IC50) is NaN
    Part4 = myMatrix.EditPart4(matrix, Part3)
    Part5 = myMatrix.EditPart5(matrix, Part4)
    Part6 = myMatrix.EditPart6(matrix, Part5)
    Part7 = myMatrix.EditPart7(matrix, Part6)
    Part8 = myMatrix.EditPart8(matrix, Part7)
    Part9 = myMatrix.EditPart9(matrix, Part8)
    Part10 = Part9
    final = myMatrix.Editfinal(matrix, Part10)
    SaveSplitFiles(final)

    def save_csv(rows, output_name):
        output_file = Path(__file__).with_name(output_name)
        with output_file.open("w", newline="", encoding="utf-8") as csv_file:
            csv.writer(csv_file).writerows(rows)
        print(f"CSV file saved to: {output_file.name}")
        return output_file

    part_files = tuple(
        save_csv(rows, f"Part{part_number}.csv")
        for part_number, rows in enumerate(
            (Part2, Part3, Part4, Part5, Part6, Part7, Part8, Part9, Part10),
            start=2,
        )
    )
    final_file = save_csv(final, "FinalCleanedData.csv")
    return part_files, final_file


if __name__ == "__main__":
    SaveMatrices("Data.xlsx")                                       ## called function
