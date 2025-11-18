import unittest
from estimator import kappa, confusion_matrix
from estimator.kappa import KappaEnum
from estimator.confusion_matrix import Matrix

class TestSampleSize(unittest.TestCase):
    def test_calc_confusion_matrix_perfect_balance(self):
        tp = 48; fp = 2; fn = 2; tn = 48
        score = Matrix(tp=tp, fp=fp, fn=fn, tn=tn).score()
        expected = 0.96
        self.assertEqual(score["f1"], expected)
        self.assertEqual(score["recall"], expected)
        self.assertEqual(score["ppv"], expected)
        self.assertEqual(score["specificity"], expected)
        self.assertTrue(score["kappa_k"] > 0.90)
        self.assertTrue(kappa.agree_near_perfect(score['kappa_k']))
        self.assertTrue(score['kappa_enum'], KappaEnum.near_perfect.name)


    def test_confusion_matrix_boundaries(self):
        f1_score = 0.95; prevalence = 0.5
        balanced = confusion_matrix.get_balanced(f1_score, prevalence)
        max_ppv = confusion_matrix.get_max_precision(f1_score, prevalence)
        max_recall = confusion_matrix.get_max_recall(f1_score, prevalence)

        # F1 harmonic balance (precision and recall)
        self.assertEqual(balanced['TP'], balanced['TN'])
        self.assertEqual(balanced['FP'], balanced['FN'])

        # optimized for precision (PPV)
        self.assertEqual(max_ppv['FP'], 0)
        self.assertTrue(max_ppv['TP'] < balanced['TN'])

        # optimized for recall (sensitivity)
        self.assertEqual(max_recall['FN'], 0)
        self.assertTrue(max_recall['TP'] > balanced['TN'])


    def test_agree_kappa_not_null(self):
        for k_int in range(0, 101):
            k_float = (k_int / 100)
            k_agree = kappa.agree_interpret(k_float)
            print('k*100', '\t', str(k_float), '\t', k_agree)
            self.assertIsNotNone(k_agree)

    def test_agree_kappa_ranges(self):
        self.assertEqual(KappaEnum.no, kappa.agree_interpret(k=0/100))
        self.assertEqual(KappaEnum.no, kappa.agree_interpret(k=1/100))
        self.assertEqual(KappaEnum.no, kappa.agree_interpret(k=9/100))

        self.assertEqual(KappaEnum.slight, kappa.agree_interpret(k=10 / 100))
        self.assertEqual(KappaEnum.slight, kappa.agree_interpret(k=20 / 100))

        self.assertEqual(KappaEnum.fair, kappa.agree_interpret(k=21 / 100))
        self.assertEqual(KappaEnum.fair, kappa.agree_interpret(k=40 / 100))

        self.assertEqual(KappaEnum.moderate, kappa.agree_interpret(k=41 / 100))
        self.assertEqual(KappaEnum.moderate, kappa.agree_interpret(k=60 / 100))

        self.assertEqual(KappaEnum.substantial, kappa.agree_interpret(k=61 / 100))
        self.assertEqual(KappaEnum.substantial, kappa.agree_interpret(k=80 / 100))

        self.assertEqual(KappaEnum.near_perfect, kappa.agree_interpret(k=81 / 100))
        self.assertEqual(KappaEnum.near_perfect, kappa.agree_interpret(k=99 / 100))

        self.assertEqual(KappaEnum.perfect, kappa.agree_interpret(k=100 / 100))