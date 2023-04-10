import test_function
import numpy as np

for i in np.arange(0.03, 0.91, 0.01):
        test_function.fingerprint_match_test (i)
        # print(i)