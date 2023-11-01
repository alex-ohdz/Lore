import json
import numpy as np
from encdec import *
from surrogate import *
from util import vector2dict
from collections import defaultdict
import copy

# Se crea una condicion para su uso en diccionarios 
def create_condition(att, op, thr, is_continuous=True):
    return {
        'att': att,
        'op': op,
        'thr': thr,
        'is_continuous': is_continuous
    }

def condition_to_string(att, op, thr, is_continuous=True):
    if is_continuous:
        if isinstance(thr, tuple):
            thr_str = f"{thr[0]} {thr[1]}"
        elif isinstance(thr, list):
            thr_str = "[" + "".join(map(str, thr)) + "]"
        else:
            thr_str = f"{thr:.2f}"
    else:
        if isinstance(thr, tuple):
            thr_str = f"[{thr[0]};{thr[1]}]"
        elif isinstance(thr, list):
            thr_str = "[" + " ; ".join(map(str, thr)) + "]"
        else:
            thr_str = f"{thr:.2f}"

    return f"{att} {op} {thr_str}"

def compare_conditions(condition1, condition2):
    # Compara los atributos 'att', 'op' y 'thr' de dos condiciones
    return (
        condition1['att'] == condition2['att'] and
        condition1['op'] == condition2['op'] and
        condition1['thr'] == condition2['thr']
    )

def are_conditions_equal(condition1, condition2):
    # Verifica si dos condiciones son iguales
    return compare_conditions(condition1, condition2)

def hash_condition(condition):
    # Convierte la condición en una cadena de texto y obtiene su hash
    return hash(str(condition))

def create_rule(premises, cons, class_name):
    return {
        'premises': premises,
        'cons': cons,
        'class_name': class_name
    }
# _pstr
def format_premises(premises):
    return '{ %s }' % (', '.join(map(str, premises)))
# pasando las premisas y la conclusión como argumentos y obtendrás la representación en cadena de la regla en el formato "premisas --> conclusión
def rule_to_string(premises, conclusion):
    premises_str = '{ %s }' % (', '.join(premises))
    return f'{premises_str} --> {conclusion}'

# comparar conjuntos de premisas y conclusiones
def compare_rules(rule1_premises, rule1_cons, rule2_premises, rule2_cons):
    premises_equal = rule1_premises == rule2_premises
    cons_equal = rule1_cons == rule2_cons
    return premises_equal and cons_equal

def get_rule_length(premises):
    return len(premises)

def my_hash_function(value):
    return hash(str(value))

def is_covered(x, feature_names, premises):
    xd = vector2dict(x, feature_names)
    for p in premises:
        op = p['op']
        att = p['att']
        thr = p['thr']
        if op == '<=' and xd[att] > thr:
            return False
        elif op == '>' and xd[att] <= thr:
            return False
    return True

def get_rule(x, y,dt, feature_names, class_name, class_values, numeric_columns, encdec=None):
    x = x.reshape(1, -1)
    feature = dt.tree_.feature
    threshold = dt.tree_.threshold

    leave_id = dt.apply(x)
    node_index = dt.decision_path(x).indices

    premises = list()
    for node_id in node_index:
        if leave_id[0] == node_id:
            break
        else:
            if encdec is not None and isinstance(encdec, OneHotEnc):
                att = feature_names[feature[node_id]]
                if att not in numeric_columns:
                    thr = 'no' if x[0][feature[node_id]] <= threshold[node_id] else 'yes'
                    op = '='
                else:
                    op = '<=' if x[0][feature[node_id]] <= threshold[node_id] else '>'
                    thr = threshold[node_id]
            else:
                op = '<=' if x[0][feature[node_id]] <= threshold[node_id] else '>'
                att = feature_names[feature[node_id]]
                thr = threshold[node_id]

            premises.append({"att": att, "op": op, "thr": thr})

    dt_outcome = dt.predict(x)[0]
    cons = class_values[int(dt_outcome)]

    return {"premises": premises, "consequence": cons, "class_name": class_name}

# obtener la profundidad maxima de un arbol 
def get_depth(dt):
    # Obtener propiedades del árbol
    n_nodes = dt.tree_.node_count
    children_left = dt.tree_.children_left
    children_right = dt.tree_.children_right

    # Inicializar la profundidad de los nodos
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    stack = [(0, -1)]  # seed is the root node id and its parent depth

    # Recorrer el árbol
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # Si el nodo actual no es un nodo hoja, agregar sus hijos a la pila
        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))

    # Obtener y devolver la máxima profundidad
    return np.max(node_depth)

# obtener las reglas del árbol 
def get_rules(dt, feature_names, class_name, class_values, numeric_columns):
    n_nodes = dt.tree_.node_count
    feature = dt.tree_.feature
    threshold = dt.tree_.threshold
    children_left = dt.tree_.children_left
    children_right = dt.tree_.children_right
    value = dt.tree_.value

    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]
    reverse_dt_dict = dict()
    left_right = dict()
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
            reverse_dt_dict[children_left[node_id]] = node_id
            left_right[(node_id, children_left[node_id])] = 'l'
            reverse_dt_dict[children_right[node_id]] = node_id
            left_right[(node_id, children_right[node_id])] = 'r'
        else:
            is_leaves[node_id] = True

    node_index_list = list()
    for node_id in range(n_nodes):
        if is_leaves[node_id]:
            node_index = [node_id]
            parent_node = reverse_dt_dict.get(node_id, None)
            while parent_node:
                node_index.insert(0, parent_node)
                parent_node = reverse_dt_dict.get(parent_node, None)
            if node_index[0] != 0:
                node_index.insert(0, 0)
            node_index_list.append(node_index)

    rules = list()
    for node_index in node_index_list:
        premises = list()
        for i in range(len(node_index) - 1):
            node_id = node_index[i]
            op = '<=' if left_right[(node_id, node_index[i+1])] == 'l' else '>'
            att = feature_names[feature[node_id]]
            thr = threshold[node_id]
            premises.append(create_condition(att, op, thr, att in numeric_columns))

        cons = class_values[np.argmax(value[node_index[-1]])]

        rule = {"premises": premises, "consequence": cons, "class_name": class_name}
        rules.append(rule)

    return rules

def compact_premises(plist):
    att_list = defaultdict(list)
    
    for p in plist:
        att_list[p['att']].append(p)

    compact_plist = list()
    for att, alist in att_list.items():
        if len(alist) > 1:
            min_thr = None
            max_thr = None
            for av in alist:
                if av['op'] == '<=':
                    max_thr = min(av['thr'], max_thr) if max_thr else av['thr']
                elif av['op'] == '>':
                    min_thr = max(av['thr'], min_thr) if min_thr else av['thr']

            if max_thr:
                compact_plist.append(create_condition(att, '<=', max_thr))

            if min_thr:
                compact_plist.append(create_condition(att, '>', min_thr))
        else:
            compact_plist.append(alist[0])
    return compact_plist

def get_counterfactual_rules(x, y, dt, Z, Y, feature_names, class_name, class_values, numeric_columns, features_map,
                             features_map_inv, encdec=None, filter_crules=None, constraints=None,
                             unadmittible_features=None):
    clen = np.inf
    crule_list = list()
    delta_list = list()
    Z1 = Z[np.where(Y != y)[0]]
    xd = vector2dict(x, feature_names)
    for z in Z1:
        crule = get_rule(z, y, dt, feature_names, class_name, class_values, numeric_columns, encdec)
        delta, qlen = get_falsified_conditions(xd, crule["premises"])
        
        if unadmittible_features:
            is_feasible = check_feasibility_of_falsified_conditions(delta, unadmittible_features)
            if not is_feasible:
                continue
        
        if constraints:
            to_remove = list()
            for p in crule['premises']:
                if p['att'] in constraints.keys():
                    if p['op'] == constraints[p['att']]['op'] and p['thr'] > constraints[p['att']]['thr']:
                        break
                    else:
                        to_remove.append(p) # Aquí falta lo que quieres hacer con `to_remove`
        
        if filter_crules:
            xc = apply_counterfactual(x, delta, feature_names, features_map, features_map_inv, numeric_columns)
            bb_outcomec = filter_crules(xc.reshape(1, -1))[0]
            bb_outcomec = class_values[bb_outcomec] if isinstance(class_name, str) else bb_outcomec # Eliminamos referencia a multilabel
            dt_outcomec = crule['cons']

            if bb_outcomec == dt_outcomec:
                if qlen < clen:
                    clen = qlen
                    crule_list = [crule]
                    delta_list = [delta]
                elif qlen == clen and delta not in delta_list:
                    crule_list.append(crule)
                    delta_list.append(delta)
        else:
            if qlen < clen:
                clen = qlen
                crule_list = [crule]
                delta_list = [delta]
            elif qlen == clen and delta not in delta_list:
                crule_list.append(crule)
                delta_list.append(delta)

    return crule_list, delta_list

def get_falsified_conditions(xd, crule_premises):
    delta = list()
    nbr_falsified_conditions = 0
    
    for p in crule_premises:
        op = p["op"]
        att = p["att"]
        thr = p["thr"]
        
        # Verificar si el atributo está en xd
        if att not in xd:
            print(f"El atributo {att} no se encuentra en xd.")
            continue

        try:
            if op == '<=' and xd[att] > thr:
                delta.append(p)
                nbr_falsified_conditions += 1
            elif op == '>' and xd[att] <= thr:
                delta.append(p)
                nbr_falsified_conditions += 1
        except Exception as e:
            print(f"Error al procesar la premisa {p}: {e}")
            continue

    return delta, nbr_falsified_conditions

def check_feasibility_of_falsified_conditions(delta, unadmittible_features):
    for p in delta:
        p_key = p["att"] if p["is_continuous"] else p["att"].split('=')[0]
        
        if p_key in unadmittible_features:
            if unadmittible_features[p_key] is None:
                return False
            else:
                if unadmittible_features[p_key] == p["op"]:
                    return False
                    
    return True

def apply_counterfactual(x, delta, feature_names, features_map=None, features_map_inv=None, numeric_columns=None):

    xd = vector2dict(x, feature_names)
    xcd = xd.copy()

    for p in delta:
        if p["att"] in numeric_columns:
            if p["thr"] == int(p["thr"]):
                gap = 1.0
            else:
                decimals = list(str(p["thr"]).split('.')[1])
                for idx, e in enumerate(decimals):
                    if e != '0':
                        break
                gap = 1 / (10**(idx+1))
            if p["op"] == '>':
                xcd[p["att"]] = p["thr"] + gap
            else:
                xcd[p["att"]] = p["thr"]
        else:
            fn = p["att"].split('=')[0]
            if p["op"] == '>':
                if features_map is not None:
                    fi = list(feature_names).index(p["att"])
                    fi = features_map_inv[fi]
                    for fv in features_map[fi]:
                        xcd['%s=%s' % (fn, fv)] = 0.0
                xcd[p["att"]] = 1.0
            else:
                if features_map is not None:
                    fi = list(feature_names).index(p["att"])
                    fi = features_map_inv[fi]
                    for fv in features_map[fi]:
                        xcd['%s=%s' % (fn, fv)] = 1.0
                xcd[p["att"]] = 0.0

    xc = np.zeros(len(xd))
    for i, fn in enumerate(feature_names):
        xc[i] = xcd[fn]

    return xc
