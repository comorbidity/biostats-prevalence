import pandas as pd
from estimator import dirichlet
import unittest

class TestProbability(unittest.TestCase):

    def test_simulation_compliance_therapeutic_trough_level(self):
        df = pd.DataFrame({
            "THERAPEUTIC": [24],
            "SUPRA_THERAPEUTIC": [3],
            "SUB_THERAPEUTIC": [2],
        })

        res = dirichlet.simulation(
            counts_df=df,
            class_cols=["THERAPEUTIC", "SUPRA_THERAPEUTIC", "SUB_THERAPEUTIC"],
            alpha=1.0,  # flat prior
            nsim=200_000,  # tighten MC error
            batch_size=25_000,  # RAM-friendly batching
            random_state=7
        )
        print(res.to_dict('records'))