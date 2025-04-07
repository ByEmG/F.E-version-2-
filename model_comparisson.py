import matplotlib.pyplot as plt
import numpy as np

# Example scores – replace with your real model results
models = ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'Support Vector Machine']
accuracy = [0.77, 0.78, 0.73, 0.79]
positive_f1 = [0.82, 0.83, 0.70, 0.84]
negative_f1 = [0.63, 0.67, 0.52, 0.68]
neutral_f1 = [0.79, 0.80, 0.54, 0.81]

x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='skyblue')
ax.bar(x - 0.5*width, positive_f1, width, label='Positive F1', color='orange')
ax.bar(x + 0.5*width, negative_f1, width, label='Negative F1', color='limegreen')
ax.bar(x + 1.5*width, neutral_f1, width, label='Neutral F1', color='tomato')

ax.set_ylabel('Score')
ax.set_xlabel('Models')
ax.set_title('Model Comparison: Key Metrics')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20)
ax.legend()

plt.tight_layout()
plt.savefig("model_comparison_graph.png")  # Saves the image
plt.show()