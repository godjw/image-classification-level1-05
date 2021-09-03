# System Libs.
from pathlib import Path

# Other Libs
import pandas as pd


def ensemble_preds(read_paths, save_path, weights=None):
    def _voting(voting):
        """
        Get voting result with voting weights.

        Args:
            voting (pd.Series): Vote for each prediction.

        Returns:
            result (int): Voting result.
        """
        result_dict = {pred_value:0 for pred_value in voting.unique()}
        for vote, weight in zip(voting, weights):
            result_dict[vote] += weight
        result = max(result_dict, key=result_dict.get)
        return result

    weights = weights if weights else [1] * len(read_paths)

    preds = []
    for f in read_paths:
        df = (pd.read_csv(f, index_col=0)).astype(int)
        preds.append(df)
    preds = pd.concat(preds, axis=1)

    ensemble_dict = dict()
    for idx, row in preds.iterrows():
        ensemble_dict[idx] = _voting(row)
    ensemble_result = (pd.DataFrame(pd.Series(ensemble_dict))).reset_index()
    ensemble_result.columns = [preds.index.name, preds.columns[0]]
    
    ensemble_result.to_csv(save_path, index=False)
    return ensemble_result




if __name__ == '__main__':
