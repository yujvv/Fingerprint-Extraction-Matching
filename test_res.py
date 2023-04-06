import test_function

# similarity_score_boundary = 500
# matching_criteria = 0.75
# block_size_std = 64
for i in (10, 800):
    for j in (100, 1300):
        test_function.fingerprint_match_test (j, 0.75, i)