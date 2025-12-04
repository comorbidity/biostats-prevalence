from math import prod

EXPLAIN_ANSWER = True

def calc_patient(patient_true:int, patient_false:int, prior_a=1.0, prior_b=1.0) -> float:
    a_true = prior_a + patient_true
    b_true = prior_b + patient_false
    return a_true/(a_true+b_true)

def calc_patient_vs_population(patient_true:int, patient_false:int, population_true:int, population_false:int, prior_a=1.0, prior_b=1.0):
    population_size = population_true + population_false
    population_prior_a = prior_a * (population_true / population_size)
    population_prior_b = prior_b * (population_false / population_size)
    prob = calc_patient(patient_true, patient_false, prior_a=population_prior_a, prior_b=population_prior_b)

    if EXPLAIN_ANSWER:
        print('#################################')
        print(f"Patient\t\t true:false\t= {patient_true}:{patient_false}")
        print(f"Population\t true:false\t= {population_true}:{population_false}")
        print(f"Posterior probability\t = {prob:.3f}")
        print('#################################')
    return prob

def calc_binomial_exact(a, b, k):
    """
    Posterior predictive probability that all k future trials are 'success'
    given posterior Beta(a, b).
    """
    terms = [(a + i) / (a + b + i) for i in range(k)]
    return prod(terms)
