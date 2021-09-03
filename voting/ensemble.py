# System Libs.
import os
from pathlib import Path

# Other Libs
import pandas as pd


def ensemble_preds(pred_folder_path, save_path, weights=None, return_result=False):
    """
    Save and get voting result from predictions.

    Args:
        pred_folder_path (list): Folder directory contains pred csv files.
        save_path (str or pathlib.Path): Voting result save path.
        weights (sequence): Weight for each prediction. Defaults to None.
        return_result (bool): Wheter return voting result.

    Returns:
        ensemble_result(pd.DataFrame): Voting result.
    """

    def _voting(voting):
        """
        Get voting result with voting weights.

        Args:
            voting (pd.Series): Vote for each prediction.

        Returns:
            result (int): Voting result.
        """
        result_dict = {pred_value: 0 for pred_value in voting.unique()}
        for vote, weight in zip(voting, weights):
            result_dict[vote] += weight
        result = max(result_dict, key=result_dict.get)
        return result

    pred_folder_path = Path(pred_folder_path)
    pred_list = os.listdir(pred_folder_path)
    pred_list = [pred_folder_path.joinpath(f) for f in pred_list if ((f[-4:] == ".csv") & (f[0] != "_"))]
    weights = weights if weights else [1] * len(pred_list)

    preds = []
    for f in pred_list:
        df = (pd.read_csv(f, index_col=0)).astype(int)
        preds.append(df)
    preds = pd.concat(preds, axis=1)

    ensemble_dict = dict()
    for idx, row in preds.iterrows():
        ensemble_dict[idx] = _voting(row)
    ensemble_result = (pd.DataFrame(pd.Series(ensemble_dict))).reset_index()
    ensemble_result.columns = [preds.index.name, preds.columns[0]]

    ensemble_result.to_csv(save_path, index=False)

    if return_result:
        return ensemble_result


if __name__ == "__main__":
    pred_folder_path = os.getcwd()
    save_path = pred_folder_path + "/" + "ensemble_result.csv"
    ensemble_preds(pred_folder_path, save_path, weights=None)
