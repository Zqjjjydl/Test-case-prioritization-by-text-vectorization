from collections import OrderedDict
import random
import time
import lsh


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def loadTestSuite(input_file, bbox=False, k=5):
    """INPUT
    (str)input_file: path of input file
    (bool)bbox: apply k-shingles to input
    (int)k: k-shingle size

    OUTPUT
    (dict)TS: key=tc_ID, val=set(covered lines/shingles)"""
    TS = {}
    with open(input_file) as fin:
        tcID = 1
        for tc in fin:
            if bbox:
                TS[tcID] = tc[:-1]
            else:
                TS[tcID] = set(tc[:-1].split())
            tcID += 1
    shuffled = list(TS.items())
    random.shuffle(shuffled)
    TS = OrderedDict(shuffled)
    if bbox:
        TS = lsh.kShingles(TS, k)
    return TS


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Jiang et al (ART-D)
# dynamic candidate set + maxmin
def art_d(input_file):
    def generate(U):
        C, T = set(), set()
        while True:
            ui = random.choice(list(U.keys()))
            S = U[ui]
            if T | S == T:
                break
            T = T | S
            C.add(ui)
        return C

    def select(TS, P, C):
        D = {}
        for cj in C:
            D[cj] = {}
            for pi in P:
                D[cj][pi] = lsh.jDistance(TS[pi], TS[cj])
        # maximum among the minimum distances
        j, jmax = 0, -1
        for cj in D.keys():
            min_di = min(D[cj].values())
            if min_di > jmax:
                j, jmax = cj, min_di
        return j

    def prioritize(TS, U, P, C):
        D = {}
        # for a in U:
        #     print(a)
        for cj in C:
            D[cj] = {}
            for pi in P:
                D[cj][pi] = lsh.jDistance(TS[pi], TS[cj])
        # maximum among the minimum distances
        j, jmax = 0, -1
        for cj in D.keys():
            min_di = min(D[cj].values())
            if min_di > jmax:
                j, jmax = cj, min_di
        return j

    # # # # # # # # # # # # # # # # # # # # # #

    ptime_start = time.process_time()

    TS = loadTestSuite(input_file)
    U = TS.copy()

    TS[0] = set()
    P = [0]

    C = generate(U)
    prioritize(TS, U, P, C)


    iteration, total = 0, float(len(U))
    while len(U) > 0:
        iteration += 1
        if iteration % 100 == 0:
            print("  Progress: {}%\r".format(
                round(100*iteration/total, 2)))

        if len(C) == 0:
            C = generate(U)
        s = select(TS, P, C)
        P.append(s)
        del U[s]
        C = C - set([s])

    ptime = time.process_time() - ptime_start

    return 0.0, ptime, P[1:]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# Zhou et al
# fixed size candidate set + manhattan distance
def art_f(input_file):
    def generate(U):
        C = set()
        if len(U) < 10:
            C = set(U.keys())
        else:
            while len(C) < 10:
                ui = random.choice(list(U.keys()))
                C.add(ui)
        return C

    def manhattanDistance(TCS, i, j):
        u, v = TCS[i], TCS[j]
        return sum([abs(float(ui) - float(vi)) for ui, vi in zip(u, v)])

    def select(TS, P, C):
        D = {}
        for cj in C:
            D[cj] = {}
            for pi in P:
                D[cj][pi] = manhattanDistance(TS, pi, cj)
        # maximum among the minimum distances
        j, jmax = 0, -1
        for cj in D.keys():
            min_di = min(D[cj].values())
            if min_di > jmax:
                j, jmax = cj, min_di

        return j

    # # # # # # # # # # # # # # # # # # # # # #

    ptime_start = time.process_time()

    TS = loadTestSuite(input_file)
    U = TS.copy()

    TS[0] = set()
    P = [0]

    C = generate(U)

    iteration, total = 0, float(len(U))
    while len(U) > 0:
        iteration += 1
        if iteration % 100 == 0:
            print("  Progress: {}%\r".format(
                round(100*iteration/total, 2)))

        if len(C) == 0:
            C = generate(U)
        s = select(TS, P, C)
        P.append(s)
        del U[s]
        C = C - set([s])

    ptime = time.process_time() - ptime_start

    return 0.0, ptime, P[1:]
