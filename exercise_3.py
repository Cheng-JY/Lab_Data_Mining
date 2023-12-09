# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Audfabe 2 ###

def accumulate(var_ration):
    acc = 0
    r = np.empty(len(var_ration))
    for idx, value in enumerate(var_ration):
        acc += value
        r[idx] = acc
    return r

def task_2():
    df = pd.read_csv('iris-data.csv')
    cov = df.cov(numeric_only=True)
    cov_num = cov.apply(pd.to_numeric, errors='coerce')
    eigenvalues = np.linalg.eigvals(cov_num)
    explained_var_ratio = eigenvalues / np.sum(eigenvalues)
    acc = accumulate(explained_var_ratio)

    #plot the diagramm
    plt.bar(range(0, len(explained_var_ratio)), explained_var_ratio, alpha=0.5,
            align='center', label='Individual explained variance')
    plt.step(range(0, len(acc)), acc, where='mid',
             label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

### Aufgabe 3 ###

def compute_impurity(feature):
    probs = feature.value_counts(normalize=True)
    impurity = -1 * np.sum(np.log2(probs) * probs)
    return round(impurity, 3)

def comp_feature_information_gain(df, descriptive_feature, target):
    print('traget feature:', target)
    print('descriptive_feature', descriptive_feature)

    target_entropy = compute_impurity(df[target])
    print('original impurity:', target_entropy)

    entropy_list = list()
    weight_list = list()

    for level in df[descriptive_feature].unique():
        df_feature_level = df[df[descriptive_feature] == level]
        entropy_level = compute_impurity(df_feature_level)
        entropy_list.append(round(entropy_level, 3))
        weight_level = len(df_feature_level) / len(df)
        weight_list.append(round(weight_level, 3))

    print('impurity of partitions: ', entropy_list)
    print('weights of partitions: ', weight_list)

    feature_remaining_impurity = np.sum(np.array(entropy_list) * np.array(weight_list))
    print('remaining impurity: ', feature_remaining_impurity)

    information_gain = target_entropy - feature_remaining_impurity
    print('information_gain: ', information_gain)

    print('===========================')

    return information_gain

def task_3():
    df = pd.read_csv('umsatz.csv')
    for feature in df.drop(columns='Umsatz').columns:
        feature_info_gain = comp_feature_information_gain(df, feature, 'Umsatz')

### Aufgabe 4 ###

def task_4():
    df = pd.read_csv('umsatz.csv')
    sample_size = len(df)
    total_count_1 = df.groupby(['Kundengruppe', 'Artikelgruppe'])['Umsatz'].aggregate('count')
    class_count_1 = df.groupby(['Kundengruppe', 'Artikelgruppe', 'Umsatz'])['Umsatz'].aggregate('count')
    result_1 = ((total_count_1-class_count_1)/total_count_1)*(total_count_1/sample_size)
    print('Kunden, Artikel: ', result_1)

    total_count_2 = df.groupby(['Kundengruppe', 'Region'])['Umsatz'].aggregate('count')
    class_count_2 = df.groupby(['Kundengruppe', 'Region', 'Umsatz'])['Umsatz'].aggregate('count')
    result_2 = ((total_count_2 - class_count_2) / total_count_2) * (total_count_2 / sample_size)
    print('Kunden, Region: ', result_2)

    total_count_3 = df.groupby(['Region', 'Artikelgruppe'])['Umsatz'].aggregate('count')
    class_count_3 = df.groupby(['Region', 'Artikelgruppe', 'Umsatz'])['Umsatz'].aggregate('count')
    result_3 = ((total_count_3 - class_count_3) / total_count_3) * (total_count_3 / sample_size)
    print('Region, Artikel: ', result_3)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    task_3()




