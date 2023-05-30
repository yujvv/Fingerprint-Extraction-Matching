import matplotlib.pyplot as plt

with open('/data.txt', 'r') as f:
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




# extract data from the dictionary
similarity_scores = list(results.keys())
false_acceptance_rates = [val['false_acceptance_rate'] for val in results.values()]
false_rejection_rates = [val['false_rejection_rate'] for val in results.values()]
error_rates = [val['error_rate'] for val in results.values()]

# create a new figure
fig, ax = plt.subplots()

# plot the data
ax.plot(similarity_scores, false_acceptance_rates, label='False Acceptance Rate')
ax.plot(similarity_scores, false_rejection_rates, label='False Rejection Rate')
ax.plot(similarity_scores, error_rates, label='Error Rate')

# add labels and legend
ax.set_xlabel('Similarity Score Threshold')
ax.set_ylabel('Discounting')
# ax.set_title('Performance Metrics by Similarity Score Boundary')
ax.legend()

# show the plot
plt.show()

# print(results)

