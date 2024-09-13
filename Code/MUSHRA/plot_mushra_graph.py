import matplotlib.pyplot as plt
import numpy as np

# Scores for each category
fx_scores = [91, 95, 91, 89, 96, 96, 97]
gen_scores = [78, 86, 88, 74, 88, 94, 85]
midq_scores = [56, 43, 62, 65, 23, 45, 58]
lowq_scores = [33, 29, 34, 55, 14, 14, 18]

# Calculating means
fx_avg = np.mean(fx_scores)
gen_avg = np.mean(gen_scores)
midq_avg = np.mean(midq_scores)
lowq_avg = np.mean(lowq_scores)

# Calculating standard deviations
fx_std = np.std(fx_scores)
gen_std = np.std(gen_scores)
midq_std = np.std(midq_scores)
lowq_std = np.std(lowq_scores)

print(f'gen_avg - {gen_avg}')
print(f'gen_std - {gen_std}')

# Labels and positions
labels = ['High Quality', 'Generated', 'Mid Quality', 'Low Quality']
x = np.arange(len(labels))
means = [fx_avg, gen_avg, midq_avg, lowq_avg]
std_devs = [fx_std, gen_std, midq_std, lowq_std]

# Creating the bar chart
plt.bar(x, means, yerr=std_devs, capsize=5, color='skyblue', edgecolor='black', alpha=0.7)
plt.xticks(x, labels)
plt.ylabel('Scores')
plt.title('Mean Scores with Standard Deviations')
plt.yticks(np.arange(0, 110, 10))

plt.grid(True)
# Display the plot
plt.show()
