import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def create_cm_plot(confusion, state):

    # Basic Metric for classification, unbalanced data though
    accuracy  = int((np.trace(confusion) / float(np.sum(confusion)))*100)
    # Metrics for Binary Confusion Matrices, better for unbalanced
    precision = (confusion[1,1] / sum(confusion[:,1]))*100
    recall    = (confusion[1,1] / sum(confusion[1,:]))*100
    f1_score  = 2*precision*recall / (precision + recall)
    stats_text = "\n\nAccuracy={:0.1f}%\nPrecision={:0.1f}%\nRecall={:0.1f}%\nF1 Score={:0.1f}%".format(
        accuracy,precision,recall,f1_score)

    # Plotting the confusion matrices
    group_names = ['True Neg', 'False Pos', 'False Neg' , 'True Pos']
    categories = ['Zero', 'One']

    group_counts = ['{:0.0f}'.format(value) for value in confusion.flatten()]

    group_percentages = ['{0:.1%}'.format(value) for value in
                        confusion.flatten()/np.sum(confusion)]

    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(
        confusion, annot=labels, fmt='', cmap='Blues', cbar=True,
        xticklabels=categories, yticklabels=categories
    )
    plt.ylabel('True label')
    plt.xlabel('Predicted label' + stats_text)
    plt.title(f'Confusion Matrix For {state} Results')
    plt.tight_layout()
    plt.savefig(f'../img/{state}/{state}_confusion_matrix.png', dpi=500, orientation='landscape');

    return None