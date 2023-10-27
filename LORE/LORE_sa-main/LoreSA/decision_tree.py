import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree._tree import TREE_LEAF


def learn_local_decision_tree(Z, Yb, weights, possibles_outcomes, cv=5,
                              poda_tree=False):
    dt = DecisionTreeClassifier()
    if poda_tree:
        param_list = {'min_samples_split': [0.002, 0.01, 0.05, 0.1, 0.2],
                      'min_samples_leaf': [0.001, 0.01, 0.05, 0.1, 0.2],
                      'max_depth': [None, 2, 4, 6, 8, 10, 12, 16]
                      }

        if len(possibles_outcomes) == 2:
            scoring = 'f1'
        else:
            scoring = 'f1_samples'


        dt_search = GridSearchCV(dt, param_grid=param_list, scoring=scoring, cv=cv, n_jobs=-1, iid=False)
        # print(datetime.datetime.now())
        dt_search.fit(Z, Yb, sample_weight=weights)
        # print(datetime.datetime.now())
        dt = dt_search.best_estimator_
        poda_duplicate_hojas(dt)
    else:
        dt.fit(Z, Yb, sample_weight=weights)

    return dt


def is_leaf(inner_tree, index):
    # Check whether node is leaf node
    return (inner_tree.children_left[index] == TREE_LEAF and
            inner_tree.children_right[index] == TREE_LEAF)


def poda_index(inner_tree, decisions, index=0):

    if not is_leaf(inner_tree, inner_tree.children_left[index]):
        poda_index(inner_tree, decisions, inner_tree.children_left[index])
    if not is_leaf(inner_tree, inner_tree.children_right[index]):
        poda_index(inner_tree, decisions, inner_tree.children_right[index])

    if (is_leaf(inner_tree, inner_tree.children_left[index]) and
        is_leaf(inner_tree, inner_tree.children_right[index]) and
        (decisions[index] == decisions[inner_tree.children_left[index]]) and
        (decisions[index] == decisions[inner_tree.children_right[index]])):
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
        # print("Pruned {}".format(index))


def poda_duplicate_hojas(dt):
    # Remove leaves if both
    decisions = dt.tree_.value.argmax(axis=2).flatten().tolist()  # Decision for each node
    poda_index(dt.tree_, decisions)
