#### Imported Libraries ####
import numpy as np 								 ##from project file
import pandas as pd								 ##from project file
from sklearn.impute import SimpleImputer 		 ##from project file
from sklearn.preprocessing import StandardScaler ##from project file

import csv
from itertools import zip_longest
from math import isfinite
from pathlib import Path
from openpyxl import Workbook, load_workbook



#### Creating Matix Object ####
class myMatrix:

        ### Reading Matrix###

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
        # Convert every value to numeric.
        # If a value is numeric:
        #       keep it
        # If a value is not numeric:
        #       convert it to NaN
        # This is done for both IC50 and the X variables.

        def EditPart2(self, original_rows):

                header = original_rows[0]										## Used to keep Header Row
                data = pd.DataFrame(original_rows[1:], columns=header).apply(	## Code from Lab Doc
                    pd.to_numeric,												##
                    errors="coerce",											##
                )																##
                print("\nNumber of NaN values in each column:")					##
                print(data.isna().sum())										##

                def clean_value(value):
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        return value

                    if not isfinite(number):
                        return np.nan
                    return int(number) if number.is_integer() else number

                original_rows = tuple(
                    tuple(clean_value(value) for value in row)
                    for row in data.itertuples(index=False, name=None)
                )

                def is_valid(values):
                    try:
                        value = float(values[0])
                        valid = isfinite(value) and value != 0.0											## Check target value and reject zeros and junk
                    except (IndexError, ValueError):
                        valid = False																		## flag if data is empty or non numeric

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
            valid_target = pd.to_numeric(data.iloc[:, 0], errors="coerce").notna()
            valid_rows = tuple(
                data.loc[valid_target].itertuples(index=False, name=None)
            )
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
            X = data.iloc[:, 1:].copy()


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
            # Here, if 99% or more of the non-NaN values are identical,
            # we consider the variable near-constant.

            near_constant_threshold = 0.525 # 52.5% near constant was the highest threshold that had 0 constant columns

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

            data = pd.concat([Y, X], axis=1)
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

            Y = data.iloc[:, 0].copy()
            X = data.iloc[:, 1:].copy()

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

                total_outliers += outliers.sum()

                # Replace outliers with NaN.
                X.loc[outliers, column] = np.nan


            print("\nOutliers in X have been converted to NaN. ", total_outliers)

            data = pd.concat([Y, X], axis=1)
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
            X = data.iloc[:, 1:].copy()

            imputer = SimpleImputer(strategy="median")

            X_imputed = imputer.fit_transform(X)

            # Convert back to a DataFrame so that we keep
            # the original X column names.
            X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

            print("\nRemaining NaN values in X:", X.isna().sum().sum())

            constant_after = [c for c in X.columns if X[c].nunique() <= 1]
            print("Constant columns after imputation:", len(constant_after), "of", X.shape[1])

            data = pd.concat([Y, X], axis=1)
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

        def EditPart9(selfself, part8):
            if not part8:
                return ()
            header = part8[0]
            data = pd.DataFrame(part8[1:], columns=header)
            Y = data.iloc[:, 0].copy()
            X = data.iloc[:, 1:].copy()

            scaler = StandardScaler()

            X_scaled_array = scaler.fit_transform(X)

            # Convert back to a DataFrame.
            X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)


            print("\nX variables have been standardized.")

            data = pd.concat([Y, X_scaled], axis=1)
            return (tuple(data.columns),) + tuple(data.itertuples(index=False, name=None))




        # ============================================================
        # 10. COMBINE Y AND CLEANED/SCALED X
        # ============================================================

        # Put the original Y (IC50) back together with
        # the cleaned and scaled X variables.

        #cleaned_data = pd.concat(
        #    [Y, X_scaled],
        #    axis=1
        #)


        # ============================================================
        # CHECK THE FINAL DATASET
        # ============================================================
        def FinalCheck(self, part9):
            if not part9:
                return ()
            header = part9[0]
            cleaned_data = pd.DataFrame(part9[1:], columns=header)

            print("\n========================================")
            print("FINAL DATASET")
            print("========================================")

            print("Number of rows:", cleaned_data.shape[0])
            print("Number of columns:", cleaned_data.shape[1])

            print("\nFirst five rows:")
            print(cleaned_data.head())

            print("\nRemaining NaN values:")
            print(cleaned_data.isna().sum().sum())

            print(cleaned_data.iloc[3, 24])


# ============================================================
# SAVE THE CLEANED DATA
# ============================================================

#cleaned_data.to_csv(
  #  "drug_data_cleaned_scaled.csv",
 #   index=False
#)

#print("\nSaved as:")
#print("drug_data_cleaned_scaled.csv")

### Print Function for Debugging ###
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

### Save and Exel File Creation ###
def SaveMatrices(file_name):
    matrix = myMatrix()                                             	## Create the Matrix class
    Part2 = myMatrix.EditPart2(matrix, matrix.GetMatrix(file_name))		## Call Part2 to remove non numerics from data
    Part3 = myMatrix.EditPart3(matrix, Part2)							## Call Part3 to remove rows where Y (IC50) is NaN
    Part4 = myMatrix.EditPart4(matrix, Part3)
    Part5 = myMatrix.EditPart5(matrix, Part4)
    Part6 = myMatrix.EditPart6(matrix, Part5)
    Part7 = myMatrix.EditPart7(matrix, Part6)
    Part8 = myMatrix.EditPart8(matrix, Part7)
    Part9 = myMatrix.EditPart9(matrix, Part8)
    myMatrix.FinalCheck(matrix, Part9)

    def save_matrix(rows, output_name, sheet_name):
        output_file = Path(__file__).with_name(output_name)
        workbook = Workbook(write_only=True)
        edited_sheet = workbook.create_sheet(sheet_name)
        for row in rows:
            edited_sheet.append(row)
        workbook.save(output_file)
        print(f"Excel workbook saved to: {output_file.name}")
        return output_file

    part2_file = save_matrix(Part2, "Part2.xlsx", "Part2")
    part3_file = save_matrix(Part3, "Part3.xlsx", "Part3")
    part4_file = save_matrix(Part4, "Part4.xlsx", "Part4")
    part5_file = save_matrix(Part5, "Part5.xlsx", "Part5")
    part6_file = save_matrix(Part6, "Part6.xlsx", "Part6")
    part7_file = save_matrix(Part7, "Part7.xlsx", "Part7")
    part8_file = save_matrix(Part8, "Part8.xlsx", "Part8")
    part9_file = save_matrix(Part9, "Part9.xlsx", "Part9")
    return part2_file, part3_file, part4_file, part5_file, part6_file, part7_file, part8_file, part9_file


if __name__ == "__main__":
    SaveMatrices("Data.xlsx")										## called function