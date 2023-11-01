from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd


# Variables globales
features = []
cont_features_names = []
cate_features_names = []
cate_features_idx = []
cont_features_idx = []
encdec = None
dataset_enc = None
onehot_feature_idx = []
new_cont_idx = []

def initialize_globals(dataset, class_name):
    global features, cont_features_names, cate_features_names, cate_features_idx
    global cont_features_idx, encdec, dataset_enc, onehot_feature_idx, new_cont_idx

    # Aquí inicializa tus variables con los datos de 'dataset' y 'class_name'

    # Ejemplo:
    features = [c for c in dataset.columns if c != class_name]

def enc_fit_transform(dataset, class_name):
    global features, cont_features_names, cate_features_names, cate_features_idx, cont_features_idx, encdec, dataset_enc
    features = [c for c in dataset.columns if c not in [class_name]]
    cont_features_names = list(dataset[features]._get_numeric_data().columns)
    cate_features_names = [c for c in dataset.columns if c not in cont_features_names and c != class_name]
    cate_features_idx = [features.index(f) for f in cate_features_names]
    cont_features_idx = [features.index(f) for f in cont_features_names]

    dataset_values = dataset[features].values
    encdec = OneHotEncoder(handle_unknown='ignore')
    dataset_enc = encdec.fit_transform(dataset_values[:, cate_features_idx]).toarray()

    onehot_feature_idx = []
    new_cont_idx = []
    for f in cate_features_idx:
        uniques = len(np.unique(dataset_values[:, f]))
        for u in range(0, uniques):
            onehot_feature_idx.append(f+u)

    npiu = i = j = 0
    while j < len(cont_features_idx):
        if cont_features_idx[j] < cate_features_idx[i]:
            new_cont_idx.append(cont_features_idx[j] + npiu - 1)
        elif cont_features_idx[j] > cate_features_idx[i]:
            npiu += len(np.unique(dataset_values[:, cate_features_idx[i]]))
            new_cont_idx.append(cont_features_idx[j] + npiu - 1)
            i += 1
        j += 1

    n_feat_tot = dataset_enc.shape[1] + len(cont_features_idx)
    dataset_enc_complete = np.zeros((dataset_enc.shape[0], n_feat_tot))
    for p in range(dataset_enc.shape[0]):
        for i in range(0, len(onehot_feature_idx)):
            dataset_enc_complete[p][onehot_feature_idx[i]] = dataset_enc[p][i]
        for j in range(0, len(new_cont_idx)):
            dataset_enc_complete[p][new_cont_idx[j]] = dataset_values[p][cont_features_idx[j]]

    return dataset_enc_complete

def enc(x, y):
    if len(x.shape) == 1:
            x_cat = x[cate_features_idx]
            x_cat = x_cat.reshape(1, -1)
            x = x.reshape(1,-1)
    else:
        x_cat = x[:, cate_features_idx]
        x_cat_enc = encdec.transform(x_cat).toarray()
        n_feat_tot = dataset_enc.shape[1] + len(cont_features_idx)
        x_res = np.zeros((x.shape[0], n_feat_tot))
        for p in range(x_res.shape[0]):
            for i in range(0, len(onehot_feature_idx)):
                x_res[p][onehot_feature_idx[i]] = x_cat_enc[p][i]
            for j in range(0, len(new_cont_idx)):
                x_res[p][new_cont_idx[j]] = x[p][cont_features_idx[j]]

    return x_res


def dec(x):
     if len(x.shape) == 1:
            x_cat = x[onehot_feature_idx]
            x = x.reshape(1, -1)
            x_cat = x_cat.reshape(1, -1)
     else:
            x_cat = x[:, onehot_feature_idx]
            #print(x_cat)
     X_new =  encdec.inverse_transform(x_cat)
     x_res = np.empty((x.shape[0], len(features)), dtype=object)
     for p in range(x.shape[0]):
            for i in range(0, len(cate_features_idx)):
                x_res[p][cate_features_idx[i]] = X_new[p][i]
            for j in cont_features_idx:
                x_res[p][j] = x[p][j]
        #print(x_res.shape)
     return x_res

# Uso
# dataset = ... # Tu DataFrame o conjunto de datos
# class_name = "tu_nombre_de_clase"
# Cargamos el CSV en un DataFrame
# dataset = pd.read_csv(r'LoreSA/datasets/iris.csv')
# # dataset = ... # Tu DataFrame o conjunto de datos
# # class_name = "tu_nombre_de_clase"
# class_name = "class"
# initialize_globals(dataset, class_name)
# encoded_data = enc_fit_transform(dataset, class_name)
# # Asumo que el nombre de la clase es 'class', cambia esto si es diferente
# # Luego, puedes llamar a tus funciones
# initialize_globals(dataset, class_name)
# encoded_data = enc_fit_transform(dataset, class_name)