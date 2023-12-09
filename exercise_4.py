import numpy.ma
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def task_2():
    data = pd.read_csv('task2.csv', header=None)
    data_copy = data.iloc[:, :-2].copy()
    print(data_copy)
    c_values = [2, 3, 5]
    for c in c_values:
        # Create a k-means model with the number of cluster
        kmeans = KMeans(
            n_clusters=c, init='random',
            n_init=1, max_iter=400,
            tol=1e-04, random_state=0)

        km = kmeans.fit(data_copy)

        plt.scatter(data_copy.iloc[:, 0], data_copy.iloc[:, 1], c=kmeans.labels_)
        plt.scatter(
            km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            s=250, marker='x',
            color='red')
        plt.title(f'Clustering with {c} clusters')
        plt.grid()
        plt.show()


def task_22():
    # Answer with Label
    data = pd.read_csv('task2.csv', header=None)
    data_copy = data.iloc[:, :-1].copy()
    data_copy.columns = ['a', 'b', 'label']
    colors = np.where(data_copy['label'] == 'Ear_right', 'r', '-')
    colors[data_copy['label'] == 'Ear_left'] = 'g'
    colors[data_copy['label'] == 'Head'] = 'b'
    data_copy.plot.scatter(x='a', y='b', c=colors)
    plt.show()

def task_3():
    xy = np.array([[0., 1., 2., 3., 4., 5.], [6., 3., 2., 1., 2., 4.]])
    corr = np.corrcoef(xy)
    print(corr)
    x = np.array([[0.], [1.], [2.], [3.], [4.], [5.]])
    y = np.array([[6.], [3.], [2.], [1.], [2.], [4.]])
    F = np.array([[1., 0., 0.], [1., 1., 1.], [1., 2., 4.], [1., 3., 9.], [1., 4., 16.], [1., 5., 25.]])
    FTF = F.T @ F
    FTF_in = np.linalg.inv(FTF)
    w = FTF_in @ F.T @ y
    print(w)

def task_4():
    x = np.array([[0.], [1.], [2.], [3.], [4.], [5.]])
    y = np.array([[6.], [3.], [2.], [1.], [2.], [4.]])
    F = np.array([[1, 9.], [1., 4.], [1., 1.], [1., 0.], [1., 1.], [1., 4.]])
    FTF = F.T @ F
    FTF_in = np.linalg.inv(FTF)
    w = FTF_in @ F.T @ y
    print(w)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    task_4()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
