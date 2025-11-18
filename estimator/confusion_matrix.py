import json
import pandas as pd
from estimator import kappa

# accuracy measures are scale invariant BUT
# NUM_SAMPLES chosen to allow for at least a small number of RARE events
NUM_SAMPLES = 1000 * 100

class Matrix(dict):
    def __init__(self, tp:int=None, fp:int=None, fn:int=None, tn:int=None):
        """
        :param tp: int True Positive
        :param fp: int False Positive
        :param fn: int False Negative
        :param tn: int True Negative
        """
        super().__init__(TP=tp, FP=fp, FN=fn, TN=tn)
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn

    def is_valid(self)-> bool:
        """
        :return: bool False if 1) infeasible negative, 2) zero POS samples, 3) zero NEG samples
        """
        if (self.tp < 0) or (self.fp < 0) or (self.fn < 0) or (self.tn < 0):
            print(f'Infeasible negative samples for confusion matrix: {self}')
            return False

        if (self.tp + self.fn) == 0:
            print(f'NO positive samples in confusion matrix {self}')
            return False

        if (self.fp + self.tn) == 0:
            print(f'NO negative samples in confusion matrix {self}')
            return False
        # checks passed (OK)
        return True

    def score(self) -> dict:
        """
        Calculate Cohen's Kappa, F1, PPV, Recall, and Precision
        from a 2x2 confusion matrix.

        Parameters:
            tp (int): True Positives (TP)
            fp (int): False Positives (FP)
            fn (int): False Negatives (FN)
            tn (int): True Negatives (TN)

        **notice** TN is LAST because it True Negatives are the least important!
        F1, Recall, and PPV dont even measure True Negatives (TN).
        The common order is tp, fp, fn, tn for that reason in accuracy calculations.
        This is non-obvious so documenting here explicitly.

        :return dict of calculated Cohen's Kappa, F1, PPV, Recall, and Precision
        """
        tp = self.tp; fp = self.fp; fn = self.fn; tn = self.tn

        if not self.is_valid():
            raise ValueError(f'Invalid confusion matrix: {self}')

        # Totals
        n = tp + fp + fn + tn
        if n == 0:
            raise ValueError("Confusion matrix is empty.")

        # Observed agreement
        po = (tp + tn) / n

        # Expected agreement
        p_yes_true = (tp + fn) / n
        p_yes_pred = (tp + fp) / n
        p_no_true = (fp + tn) / n
        p_no_pred = (fn + tn) / n
        pe = p_yes_true * p_yes_pred + p_no_true * p_no_pred

        kappa_k = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # F1-score
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        return {
            "kappa_k": round(kappa_k, 3),
            "kappa_enum": str(kappa.agree_interpret(kappa_k).name),
            "f1": round(f1, 3),
            "ppv": round(precision, 3),  # synonym
            "recall": round(recall, 3),
            "specificity": round(specificity, 3)}

#####################################################################################################################
# Get Confusion Matrix
#####################################################################################################################
def get_balanced(f1_score:float=0.95, prevalence:float=0.5, n:int=NUM_SAMPLES) -> Matrix:
    """
    Balanced: perfect F1 balance means Precision = Recall = F1
    """
    positives = int(n * prevalence)
    negatives = n - positives

    tp = int(round(f1_score * positives))
    fn = positives - tp
    fp = int(round(tp * (1 - f1_score) / f1_score))
    tn = negatives - fp

    return Matrix(tp, fp, fn, tn)

def get_max_precision(f1_score:float=0.95, prevalence:float=0.5, n:int=NUM_SAMPLES) -> Matrix:
    """
    Max precision: Precision = 1.0, solve Recall from F1
    """
    positives = int(n * prevalence)
    negatives = n - positives

    recall = f1_score / (2 - f1_score)  # solved above
    tp = int(round(recall * positives))
    fn = positives - tp
    fp = 0  # precision = 1 means no false positives
    tn = negatives - fp

    return Matrix(tp, fp, fn, tn)

def get_max_recall(f1_score: float = 0.95, prevalence: float = 0.5, n: int = NUM_SAMPLES) -> Matrix:
    """
    Max recall given F1 and prevalence, with a valid confusion matrix.

    We fix:
      - F1 (f1_score)
      - prevalence (π = positives / n)
      - total n

    and we choose the largest feasible recall R such that:
      - F1 = 2 * P * R / (P + R)
      - TN >= 0, FP <= negatives
      - 0 < P <= 1

    In many cases, R can be 1.0; when that would make TN < 0, we cap R
    at the maximum feasible value implied by F1 and prevalence.
    """
    f = float(f1_score)
    pi = float(prevalence)

    if not (0 < f < 1):
        raise ValueError(f"f1_score must be in (0, 1), got {f}")
    if not (0 < pi < 1):
        raise ValueError(f"prevalence must be in (0, 1), got {pi}")

    positives = int(round(n * pi))
    negatives = n - positives
    if positives == 0 or negatives < 0:
        raise ValueError(f"Infeasible n/prevalence combo: positives={positives}, negatives={negatives}")

    # Condition for being able to have recall = 1 with F1 = f at prevalence = pi:
    #   pi <= f / (2 - f)
    # If that's true, we can push recall all the way to 1.
    recall_upper_if_possible = f / (2 - f)
    if pi <= recall_upper_if_possible:
        recall = 1.0
    else:
        # Otherwise, constraint TN >= 0 caps recall:
        #   R_max = F / (pi * (2 - F))
        recall = f / (pi * (2 - f))

    # Precision from F1 and recall:
    #   F = 2 P R / (P + R)  =>  P = F R / (2R - F)
    denom = (2 * recall - f)
    if denom <= 0:
        raise ValueError(f"Infeasible (f1={f}, prevalence={pi}) recall={recall}")
    precision = f * recall / denom

    # Now build counts
    tp = int(round(recall * positives))
    fn = positives - tp

    # FP from precision = TP / (TP + FP)
    fp = int(round(tp * (1 - precision) / precision)) if precision > 0 else negatives
    tn = negatives - fp

    # Small rounding slop fix: if tn is slightly negative (e.g. -1), clamp.
    if tn < 0:
        # If it's more than a small rounding artifact, treat as error.
        if tn < -1:
            raise ValueError(f"Discrete rounding made TN < 0 (TN={tn}) for f1={f}, prevalence={pi}, n={n}")
        # Adjust FP/TN minimally
        fp += tn  # tn is negative, so this reduces fp
        tn = 0

    return Matrix(tp, fp, fn, tn)

def json_to_csv(filename_json:str):
    with open(filename_json, "r") as f:
        content = json.load(f)
    df = pd.json_normalize(content)
    df.columns = df.columns.str.replace(r"\.", "_", regex=True)
    return df.to_csv(f'{filename_json}.csv', index=False)


def simulate():
    """
    Create an output file (JSON or CSV) of sample size requirements for given F1 accuracy and % prevalence of phenotype
    """
    output = list()
    for f1_score in [0.80, 0.85, 0.9, 0.95, 0.99, 0.999]:
        for prevalence in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]:
            if f1_score >= prevalence:
                balanced = get_balanced(f1_score, prevalence)
                max_precision  = get_max_precision(f1_score, prevalence)
                max_recall = get_max_recall(f1_score, prevalence)

                output.append({
                    'f1':f1_score,
                    'prevalence':prevalence,
                    'balanced': {'matrix': balanced, 'score': balanced.score()},
                    'max_precision': {'matrix': max_precision, 'score': max_precision.score()},
                    'max_recall': {'matrix': max_recall, 'score': max_recall.score()},
                })
    print(output)
    with open('confusion_matrix.json', 'w') as f:
        json.dump(output, f, indent=4)
    json_to_csv('confusion_matrix.json')


if __name__ == '__main__':
    simulate()