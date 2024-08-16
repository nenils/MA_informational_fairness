from collections import defaultdict

import numpy as np
import scipy as sp
import pandas as pd


# Function to calculate the sparsity score
def sparsity(df_cf, df_cf_found, df_fc_found):

    # If there's counterfactual result
    if df_cf_found.shape[0] > 0:

        if 'Loan_Status' in df_fc_found.columns:
            # Copy the found CF results
            df_fc_found = df_fc_found.copy().drop(columns=['Loan_Status'])
            df_cf_found = df_cf_found.copy().drop(columns=['Loan_Status'])
        else:
            df_fc_found = df_fc_found.copy()
            df_cf_found = df_cf_found.copy()

        # Create an array to store output results
        scores = []

        # For each result index
        for i in range(df_cf_found.shape[0]):
            # Calculate the number of changes by the total number of features (sparsity)
            scores.append((df_cf_found.iloc[i].round(4) == df_fc_found.iloc[i].round(4)).sum() / df_cf_found.shape[1])

        # Transfer the result to the standard output variable
        out_array = scores

        # Create the result output array
        output_results = [np.nan] * df_cf.shape[0]

        output_results = dict(zip(list(df_fc_found.index), scores)) 


        return output_results

    return [np.nan] * df_cf.shape[0]


# Returns the coverage (validity) score, that inform if a CF indeed flipped the class result
def validity_total(df_cf, df_fc, model):
    # (not_is_na) and ((cf > 0.5)!=(fc > 0.5) or cf==0.5)

    # Create a copy of the counterfactuals and factuals
    df_cf = df_cf.copy().drop(columns=['Loan_Status'])
    df_fc = df_fc.copy().drop(columns=['Loan_Status'])


    not_is_na = df_cf.isna().sum(axis=1) == 0



    pred_cf = model.predict(df_cf)
    pred_cf = pred_cf[0]
    cf = pd.Series(pred_cf.reshape(-1))
    pred_fc = model.predict(df_fc)
    pred_fc = pred_fc[0]
    fc = pd.Series(pred_fc.reshape(-1))

    print(fc)
    print(cf)

    
    return (not_is_na & (((cf > 0.5) != (fc > 0.5)) | (cf.apply(lambda x: round(x, 5)) == 0.5))).to_list()


# Calculates the Mean Absolute Deviation Distance
def madd(df_oh, df_cf, num_columns, cat_columns, df_cf_found, df_fc_found):

    # If there are counterfactuals
    if df_cf_found.shape[0] > 0:

        # Create an empty frame to store the results
        df_mad = df_cf_found.iloc[:0]

        # Create a copy of the df_oh frame, avoiding to alterate the original
        df_oh_c = df_oh.copy()

        # Get only the found CFs
        df_oh_c.columns = df_fc_found.columns

        # Create a dictionary to store the MAD distance for each result
        mad_num = {}
        for n_feat_idx in num_columns:
            # 1e-8 added to avoid 0 and, then, division by zero
            mad_num[n_feat_idx] = sp.stats.median_abs_deviation(df_oh_c.loc[:, n_feat_idx]) + 1e-8

            # Calculate the distance using the MAD
            df_mad[n_feat_idx] = abs(df_cf_found[n_feat_idx] - df_fc_found[n_feat_idx]) / mad_num[n_feat_idx]

        for c_feat_idx in cat_columns:
            # If it's a categorical feature, we use 1 distance if it's different and 0 if the same
            df_mad[c_feat_idx] = (df_cf_found[c_feat_idx] != df_fc_found[c_feat_idx]).map(int)

        # Create an array to output the result
        output_result = [0] * df_cf.shape[0]

        print(df_mad)

        # If there are categorical features
        if len(cat_columns) > 0:
            # Get the mean result for the categorical features distances
            add_output_result = df_mad[cat_columns].mean(axis=1)
            # For those rows that did not generate CF results

            #for null_row in list(set([*range(len(output_result))]) - set(df_mad.index)):
            #   add_output_result.loc[null_row] = np.nan

            # Sort by the index order
            #add_output_result = add_output_result.sort_index()

            add_output_result = add_output_result.values.flatten()
            
            # Sum the results to the output array
            output_result = np.add(output_result,add_output_result.tolist())

        # If there are numerical features
        if len(num_columns) > 0:

            # Get the mean reasult for the numerical features distances
            add_output_result = df_mad[num_columns].mean(axis=1)
            print(add_output_result)
            # For those rows that did not generate CF results
            #for null_row in list(set([*range(len(output_result))]) - set(df_mad.index)):
            #    add_output_result.loc[null_row] = np.nan

            # Sort by the index order
            #add_output_result = add_output_result.sort_index()
            add_output_result = add_output_result.values.flatten()

            print(add_output_result)

            # Sum the results to the output array
            output_result = np.add(output_result, add_output_result.tolist())

        # Convert the output result frame to a list
        out_array = output_result.tolist()

        print(out_array)

        # Create a final output array with NAN values
        output_results = [np.nan] * df_cf.shape[0]

        # Replace the NaN values for results when there's a found CF result
        output_results = dict(zip(list(df_fc_found.index), out_array))
        return output_results

    return [np.nan] * df_cf.shape[0]


# Mahalanobis Distance metric calculation
def md(df_oh, df_cf, df_cf_found, df_fc_found):

    # If there are counterfactuals
    if df_cf_found.shape[0] > 0:

        # Create array to store the results
        output_result = []

        # For each row index
        for idx in range(df_cf_found.shape[0]):
            # Calculate the mahalanobis distance between the counterfactual and factual and
            # having the dataset covariance matrix
            m_dis = sp.spatial.distance.mahalanobis(df_cf_found.iloc[idx].to_numpy(),
                                                    df_fc_found.drop(columns=['Loan_Status']).iloc[idx].to_numpy(),
                                                    df_oh.drop(columns=['Loan_Status']).cov().to_numpy())
            # Append row results
            output_result.append(m_dis)

        # Store the result in another variable to mantain pattern
        out_array = output_result

        # Create a NaN array with the same number of CF results
        output_results = [np.nan] * df_cf.shape[0]

        output_results = dict(zip(list(df_fc_found.index), out_array)) 

        return output_results

    return [np.nan] * df_cf.shape[0]


# This result check if categorical features follow the binarization rule
def check_binary_categorical(df_cf, cat_columns):
    df_not_nan = df_cf.isna().sum(axis=1) == 0
    df_has_bin_values = ((df_cf.loc[:, cat_columns] != 1) & (df_cf.loc[:, cat_columns] != 0)).sum(axis=1) == 0

    return (df_not_nan & df_has_bin_values).to_numpy()


# Verifies if the one-hot encoded features only activated one feature
def check_one_hot_integrity(df_cf, cat_columns):
    # Group columns with the same prefix
    cat_groups = defaultdict(list)
    for col in cat_columns:
        cat_groups[col.split('_')[0]].append(col)
    df_not_nan = df_cf.isna().sum(axis=1) == 0
    check_groups = [df_cf.loc[:, group_values].sum(axis=1) == 1 for group_values in cat_groups.values()
                    if len(group_values) > 1]
    df_ohe_integrity = sum(check_groups) == len(check_groups)

    return (df_not_nan & df_ohe_integrity).to_numpy()




# Get the Information about the Counterfactual distances to the factuals 

from typing import List

import numpy as np
import pandas as pd


def l0_distance(delta: np.ndarray) -> List[float]:
    """
    Computes L-0 norm, number of non-zero entries.

    Parameters
    ----------
    delta: np.ndarray
        Difference between factual and counterfactual

    Returns
    -------
    List[float]
    """
    # get mask that selects all elements that are NOT zero (with some small tolerance)
    difference_mask = np.invert(np.isclose(delta, np.zeros_like(delta), atol=1e-05))
    # get the number of changed features for each row
    num_feature_changes = np.sum(
        difference_mask,
        axis=1,
        dtype=np.float,
    )
    distance = num_feature_changes.tolist()
    return distance


def l1_distance(delta: np.ndarray) -> List[float]:
    """
    Computes L-1 distance, sum of absolute difference.

    Parameters
    ----------
    delta: np.ndarray
        Difference between factual and counterfactual

    Returns
    -------
    List[float]
    """
    absolute_difference = np.abs(delta)
    distance = np.sum(absolute_difference, axis=1, dtype=np.float).tolist()
    return distance


def l2_distance(delta: np.ndarray) -> List[float]:
    """
    Computes L-2 distance, sum of squared difference.

    Parameters
    ----------
    delta: np.ndarray
        Difference between factual and counterfactual

    Returns
    -------
    List[float]
    """
    squared_difference = np.square(np.abs(delta))
    distance = np.sum(squared_difference, axis=1, dtype=np.float).tolist()
    return distance


def linf_distance(delta: np.ndarray) -> List[float]:
    """
    Computes L-infinity norm, the largest change

    Parameters
    ----------
    delta: np.ndarray
        Difference between factual and counterfactual

    Returns
    -------
    List[float]
    """
    absolute_difference = np.abs(delta)
    # get the largest change per row
    largest_difference = np.max(absolute_difference, axis=1)
    distance = largest_difference.tolist()
    return distance


def _get_delta(factual: np.ndarray, counterfactual: np.ndarray) -> np.ndarray:
    """
    Compute difference between original factual and counterfactual

    Parameters
    ----------
    factual: np.ndarray
        Normalized and encoded array with factual data.
        Shape: NxM
    counterfactual: : np.ndarray
        Normalized and encoded array with counterfactual data.
        Shape: NxM

    Returns
    -------
    np.ndarray
    """
    return counterfactual - factual


def _get_distances(
    factual: np.ndarray, counterfactual: np.ndarray
) -> List[List[float]]:
    """
    Computes distances.
    All features have to be in the same order (without target label).

    Parameters
    ----------
    factual: np.ndarray
        Normalized and encoded array with factual data.
        Shape: NxM
    counterfactual: np.ndarray
        Normalized and encoded array with counterfactual data
        Shape: NxM

    Returns
    -------
    list: distances 1 to 4
    """
    if factual.shape != counterfactual.shape:
        raise ValueError("Shapes of factual and counterfactual have to be the same")
    if len(factual.shape) != 2:
        raise ValueError(
            "Shapes of factual and counterfactual have to be 2-dimensional"
        )

    # get difference between original and counterfactual
    delta = _get_delta(factual, counterfactual)

    d1 = l0_distance(delta)
    d2 = l1_distance(delta)
    d3 = l2_distance(delta)
    d4 = linf_distance(delta)

    return [[d1[i], d2[i], d3[i], d4[i]] for i in range(len(d1))]


class Distance():
    """
    Calculates the L0, L1, L2, and L-infty distance measures.
    """

    def __init__(self):
        self.columns = ["L0_distance", "L1_distance", "L2_distance", "Linf_distance"]

    def get_evaluation(self, factuals, counterfactuals):

        factuals = factuals.drop(columns=["Loan_Status"])
        counterfactuals = counterfactuals.drop(columns=["Loan_Status"])


        # only keep the rows for which counterfactuals could be found
        counterfactuals_without_nans, factuals_without_nans = counterfactuals, factuals
        
        

        # return empty dataframe if no successful counterfactuals
        if counterfactuals_without_nans.empty:
            return pd.DataFrame(columns=self.columns)

        arr_f = (factuals_without_nans).to_numpy()
        arr_cf = (counterfactuals_without_nans).to_numpy()

        distances = _get_distances(arr_f, arr_cf)

        return pd.DataFrame(distances, columns=self.columns)