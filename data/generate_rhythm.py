import numpy as np


"""This file is used to subdivide measures into smaller units.
Everything is with respect to 16th notes, a 4/4 measure is 16 units long. If it is divided into (2, 2, 4, 4, 4)
then this corresponds to two eighth notes followed by three quarter notes. The division is done recursively,
and at every step there is a probability to not subdivide, and probability to split the division into 2.
These probabilities are encoded in each of the values of the dictionary tp.
The different values in tp correspond to different ways to divide the measure up.
For example tp['quarters'] only divides the measure into quarters.
"""


tp = dict()
tp['complex'] = dict()
tp['complex'][16] = {None: 15,
        (1, 15): 5,
        (2, 14): 2.5,
        (3, 13): 2.5,
        (4, 12): 50,
        (5, 11): 2.5,
        (6, 10): 5,
        (7, 9): 2.5,
        (8, 8): 100}

tp['complex'][15] = {(1, 14): 1,
        (2, 13): 1,
        (3, 12): 1,
        (4, 11): 2,
        (5, 10): 1,
        (6, 9): 5,
        (7, 8): 1}

tp['complex'][14] = {(1, 13): 1,
        (2, 12): 20,
        (3, 11): 1,
        (4, 10): 2,
        (5, 9): 1,
        (6, 8): 20,
        (7, 7): 1}

tp['complex'][13] = {(1, 12): 2,
        (2, 11): 1,
        (3, 10): 2,
        (4, 9): 1,
        (5, 8): 1,
        (6, 7): 1}

tp['complex'][12] = {None: 35,
        (1, 11): 1,
        (2, 10): 1,
        (3, 9): 20,
        (4, 8): 100,
        (5, 7): 1,
        (6, 6): 100}

tp['complex'][11] = {(1, 10): 1,
        (2, 9): 1,
        (3, 8): 10,
        (4, 7): 2,
        (5, 6): 5}

tp['complex'][10] = {(1, 9): 1,
        (2, 8): 50,
        (3, 7): 5,
        (4, 6): 10,
        (5, 5): 1}

tp['complex'][9] = {(1, 8): 1,
        (2, 7): 1,
        (3, 6): 5,
        (4, 5): 1}

tp['complex'][8] = {None: 50,
        (1, 7): 1,
        (2, 6): 10,
        (3, 5): 1,
        (4, 4): 50}

tp['complex'][7] = {(1, 6): 1,
        (2, 5): 1,
        (3, 4): 10}

tp['complex'][6] = {None: 25,
        (1, 5): 1,
        (2, 4): 100,
        (3, 3): 10}

tp['complex'][5] = {(1, 4): 10,
        (2, 3): 2}

tp['complex'][4] = {None: 200,
        (1, 3): 5,
        (2, 2): 75}

tp['complex'][3] = {None: 50,
        (1, 2): 10}

tp['complex'][2] = {None: 100,
        (1, 1): 10}

tp['complex'][1] = {None: 1}

tp['wholes'] = dict()
tp['wholes'][16] = {None: 1}
tp['wholes'][12] = {None: 1}
tp['wholes'][8] = {None: 1}

tp['halves_wholes'] = dict()
tp['halves_wholes'][16] = {None: 10, (8, 8): 40}
tp['halves_wholes'][12] = {None: 10, (8, 4): 40}
tp['halves_wholes'][8] = {None: 3, (4, 4): 1}
tp['halves_wholes'][4] = {None: 1}

tp['quarters'] = dict()
tp['quarters'][16] = {(4, 4, 4, 4): 1}
tp['quarters'][12] = {(4, 4, 4): 1}
tp['quarters'][8] = {(4, 4): 1}
tp['quarters'][4] = {None: 1}

tp['eighths'] = dict()
tp['eighths'][16] = {(2, 2, 2, 2, 2, 2, 2, 2): 1}
tp['eighths'][12] = {(2,)*6: 1}
tp['eighths'][8] = {(2,)*4: 1}
tp['eighths'][2] = {None: 1}

tp['sixteenths'] = dict()
tp['sixteenths'][16] = {(1,)*16: 1}
tp['sixteenths'][12] = {(1,)*12: 1}
tp['sixteenths'][8] = {(1,)*8: 1}
tp['sixteenths'][1] = {None: 1}

tp['quarters_eighths'] = dict()
tp['quarters_eighths'][16] = {(4, 4, 4, 4): 1}
tp['quarters_eighths'][12] = {(4, 4, 4): 1}
tp['quarters_eighths'][8] = {(4, 4): 1}
tp['quarters_eighths'][4] = {None: 1, (2, 2): 1}
tp['quarters_eighths'][2] = {None: 1}

tp['quarters_eighths_sixteenths'] = dict()
tp['quarters_eighths_sixteenths'][16] = {(4, 4, 4, 4): 1}
tp['quarters_eighths_sixteenths'][12] = {(4, 4, 4): 1}
tp['quarters_eighths_sixteenths'][8] = {(4, 4): 1}
tp['quarters_eighths_sixteenths'][4] = {None: 2, (2, 2): 1, (1, 1, 1, 1): 1}
tp['quarters_eighths_sixteenths'][2] = {None: 5, (1, 1): 1}
tp['quarters_eighths_sixteenths'][1] = {None: 1}

tp['halves_quarters_eighths_sixteenths'] = dict()
tp['halves_quarters_eighths_sixteenths'][16] = {(12, 4): 1, (8, 4, 4): 3, (4, 8, 4): 1, (4, 4, 8): 3, (4, 12): 1}
tp['halves_quarters_eighths_sixteenths'][12] = {None: 2, (8, 4): 1, (4, 8): 1}
tp['halves_quarters_eighths_sixteenths'][8] = {None: 4, (4, 4): 1}
tp['halves_quarters_eighths_sixteenths'][4] = {None: 2, (2, 2): 1, (1, 1, 1, 1): 1}
tp['halves_quarters_eighths_sixteenths'][2] = {None: 5, (1, 1): 1}
tp['halves_quarters_eighths_sixteenths'][1] = {None: 1}

tp['quarters_eighths_sixteenths_dots'] = dict()
tp['quarters_eighths_sixteenths_dots'][16] = {(6, 2, 4, 4): 1,
                                              (2, 6, 4, 4): 1,
                                              (4, 4, 6, 2): 1,
                                              (4, 4, 2, 6): 1,
                                              (4, 6, 2, 4): 1,
                                              (4, 2, 6, 4): 1,
                                              (4, 4, 4, 4): 3}
tp['quarters_eighths_sixteenths_dots'][12] = {(6, 2, 4): 1, (4, 6, 2): 1, (2, 6, 4): 1, (4, 2, 6): 1, (4, 4, 4): 4}
tp['quarters_eighths_sixteenths_dots'][8] = {(6, 2): 3, (2, 6): 1, (4, 4): 4}
tp['quarters_eighths_sixteenths_dots'][6] = {None: 1}
tp['quarters_eighths_sixteenths_dots'][4] = {None: 12, (2, 2): 3, (3, 1): 12, (1, 3): 5}
tp['quarters_eighths_sixteenths_dots'][3] = {None: 1}
tp['quarters_eighths_sixteenths_dots'][2] = {None: 1}
tp['quarters_eighths_sixteenths_dots'][1] = {None: 1}




def normalize_tp(tp):
    # normalize transition probabilities to be actual probabilities
    new = dict()
    for div in tp:
        new[div] = dict()
        for subdiv in tp[div]:
            if subdiv and subdiv[0] != subdiv[1]:
                new[div][(subdiv[1], subdiv[0])] = tp[div][subdiv]

    for div in tp:
        tp[div] = {**tp[div], **new[div]}

    for div in tp:
        total = np.sum([tp[div][subdiv] for subdiv in tp[div]])
        for subdiv in tp[div]:
            tp[div][subdiv] /= total

    return tp

for x in tp:
    tp[x] = normalize_tp(tp[x])
    

    
def generate_rhythm(tp_key, measure_length):
    # measure length is 16, 12, or 8 for 4/4, 3/4, or 2/4
    # tp_key is a key of the dictionary tp defined at this file
    # returns a random artition of measure length as a list,
    # for example if measure_length is 12 it might return [4, 4, 1, 3] 
    items = list(tp[tp_key][measure_length].items())
    probs = [prob for _, prob in items]
    n = np.random.choice(len(probs), p=probs)
    split = items[n][0]
    if not split:
        return [measure_length]
    else:
        pieces = [generate_rhythm(tp_key, x) for x in split]
        subdivision = [x for y in pieces for x in y]
        return subdivision
    
    
for key in tp:
    print(key, generate_rhythm(key, np.random.choice([8, 12, 16])))