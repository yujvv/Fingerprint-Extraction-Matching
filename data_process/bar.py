import matplotlib.pyplot as plt

with open('data.txt', 'r') as f:
    lines = f.readlines()

results = {}

for i in range(0, len(lines), 7):
    similarity_score_boundary = float(lines[i].split(': ')[1].strip())
    successful_matches = int(lines[i+2].split(': ')[1].strip())
    far = float(lines[i+3].split(': ')[1].strip())
    frr = float(lines[i+4].split(': ')[1].strip())
    err = float(lines[i+5].split(': ')[1].strip())

    results[similarity_score_boundary] = {
        'successful_matches': successful_matches,
        'false_acceptance_rate': far,
        'false_rejection_rate': frr,
        'error_rate': err
    }


# Data
similarity_scores = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25]
false_acceptance_rates = [results[score]['false_acceptance_rate'] for score in similarity_scores]
false_rejection_rates = [results[score]['false_rejection_rate'] for score in similarity_scores]
error_rates = [results[score]['error_rate'] for score in similarity_scores]

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(similarity_scores, false_acceptance_rates, label='False Acceptance Rate')
ax.bar(similarity_scores, false_rejection_rates, bottom=false_acceptance_rates, label='False Rejection Rate')
ax.bar(similarity_scores, error_rates, bottom=[sum(x) for x in zip(false_acceptance_rates, false_rejection_rates)], label='Error Rate')

# Axis labels and legend
ax.set_xlabel('Similarity Score Boundaries')
ax.set_ylabel('Percentage')
ax.legend()

# Show plot
plt.show()
