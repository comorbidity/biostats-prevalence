import unittest
from cumulus_library_kidney_transplant import jaccard

class TestJaccard(unittest.TestCase):

    def test_identity(self):
        # Example usage
        A = {1, 2, 3}
        B = {2, 3, 4, 5}

        expected = 0.4
        actual = jaccard.score(A, B)
        self.assertEqual(expected, actual)

    def test_example(self):
        """
        https://people.revoledu.com/kardi/tutorial/Similarity/Jaccard.html
        :return:
        """
        A = {7, 3, 2, 4, 1}
        B = {4, 1, 9, 7, 5}

        expected = 0.429
        actual = jaccard.score(A, B)
        self.assertEqual(expected, actual)
