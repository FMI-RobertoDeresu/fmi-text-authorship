import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA


def plot(data):
    features, labels = (np.array(data[0]), np.array(data[1]))

    col_sums = features.sum(axis=0)
    features_norm_cols = features / col_sums[np.newaxis, :]

    pca = sklearnPCA(n_components=2)  # 2-dimensional LDA
    pca_transformed = pd.DataFrame(pca.fit_transform(features_norm_cols))

    plt.figure(1)
    plt.scatter(pca_transformed[labels == 0][0], pca_transformed[labels == 0][1], label='HAMILTON', c='red')
    plt.scatter(pca_transformed[labels == 1][0], pca_transformed[labels == 1][1], label='MADISON', c='blue')
    plt.show()
