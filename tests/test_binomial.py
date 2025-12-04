import unittest
from estimator import binomial

class TestBinomial(unittest.TestCase):

    def test_kidney_donor_related(self):
        # Example DonorRelated=True
        patient_true = 6
        patient_false = 0
        population_true = 43
        population_false = 19

        for trials in range(patient_true+1):
            prob = binomial.calc_patient_vs_population(trials, patient_false, population_true, population_false)
            accept = (prob >= 0.95) or (prob < 0.05)

            if trials < patient_true:
                self.assertFalse(accept)
            else:
                self.assertTrue(accept)

    def test_kidney_donor_unrelated(self):
        # Example DonorRelated=False
        patient_true = 0
        patient_false = 13
        population_true = 43
        population_false = 19

        for trials in range(patient_false+1):
            prob = binomial.calc_patient_vs_population(patient_true, trials, population_true, population_false)
            accept = (prob >= 0.95) or (prob < 0.05)

            if trials < patient_false:
                self.assertFalse(accept)
            else:
                self.assertTrue(accept)
