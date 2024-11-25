import ast
import random


def tj_class(TJ):
    tj0r = [0 for i in TJ.get("tj0r")]
    tj1r = [1 for i in TJ.get("tj1r")]
    tj2r = [2 for i in TJ.get("tj2r")]
    tj3r = [3 for i in TJ.get("tj3r")]

    TJ = tj0r + tj1r + tj2r + tj3r
    random.shuffle(TJ)

    return  TJ


def string_to_dict(input_str):
    result_dict = ast.literal_eval(input_str)
    if isinstance(result_dict, dict):
        return result_dict

def Graph(adj, R):
    ad = []
    for k in adj.keys():
        row = []
        for kk in adj.get(k).keys():
            row.append(adj.get(k).get(kk))

        if k in R :
            row = [i * 3 for i in row]

        ad.append(row)

    return ad

