from ENCODECP import *

def initialize_lorem(K, bb_predict, predict_proba, feature_names, class_name, class_values, numeric_columns, 
                     features_map, neigh_type='genetic', K_transformed=None, categorical_use_prob=True, 
                     continuous_fun_estimation=False, size=1000, ocr=0.1, one_vs_rest=False, filter_crules=True,
                     init_ngb_fn=True, kernel_width=None, kernel=default_kernel, random_state=None, 
                     ENCODECP=None, dataset=None, binary=False, discretize=True, verbose=False, 
                     extreme_fidelity=False, constraints=None, **kwargs):

    lorem_properties = {}

    lorem_properties["random_state"] = random_state
    lorem_properties["bb_predict"] = bb_predict
    lorem_properties["predict_proba"] = predict_proba
    lorem_properties["class_name"] = class_name
    lorem_properties["unadmittible_features"] = None
    lorem_properties["feature_names"] = feature_names
    lorem_properties["class_values"] = class_values
    lorem_properties["numeric_columns"] = numeric_columns
    lorem_properties["features_map"] = features_map
    lorem_properties["neigh_type"] = neigh_type
    lorem_properties["one_vs_rest"] = one_vs_rest
    lorem_properties["filter_crules"] = bb_predict if filter_crules else None
    lorem_properties["binary"] = binary
    lorem_properties["verbose"] = verbose
    lorem_properties["discretize"] = discretize
    lorem_properties["extreme_fidelity"] = extreme_fidelity
    

    if ENCODECP == 'onehot':
        print('preparo onehotencoding')
        ENCODECP = OneHotEnc(dataset, class_name)
        ENCODECP.enc_fit_transform()
        Y = bb_predict(K)
        print('la y calcolata ', Y)
        K = ENCODECP.enc(K, Y)
    else:
        ENCODECP = None

    K_original = K_transformed

    if features_map:
        features_map_inv = dict()
        for idx, idx_dict in features_map.items():
            for k, v in idx_dict.items():
                features_map_inv[v] = idx

    kernel_width = np.sqrt(len(feature_names)) * .75 if kernel_width is None else kernel_width
    kernel = partial(kernel, kernel_width=kernel_width)

    np.random.seed(random_state)

    if init_ngb_fn:
        init_neighbor_fn(ocr, categorical_use_prob, continuous_fun_estimation, size, kwargs)  # Esta función aún debe ser definida fuera de la clase

    return {
        "K": K,
        "features_map_inv": features_map_inv,
        "kernel": kernel,
	}