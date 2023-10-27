import json
import numpy as np
from encdec import *
from surrogate import *
from util import vector2dict, multilabel2str
from collections import defaultdict


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

            premises.append((att, op, thr))

    dt_outcome = dt.predict(x)[0]
    cons = class_values[int(dt_outcome)]

    return {"premises": premises, "consequence": cons, "class_name": class_name}
