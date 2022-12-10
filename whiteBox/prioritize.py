import competitors
import metric

usage = """USAGE: python py/prioritize.py <dataset> <entity> <algorithm> <repetitions>
OPTIONS:
  <dataset>: test suite to prioritize.
    options: closure_v0, lang_v0, math_v0, chart_v0, time_v0
  <entity>: use function/branch/line coverage information.
    options: function, branch, line
  <algorithm>: algorithm used for prioritization.
    options: ART-D, ART-F
  <repetitions>: number of prioritization to compute.
    options: positive integer value, e.g. 50
"""

def wboxPrioritization(name, prog, v, ctype, repeats):

    fin = "../input/{}_{}/{}-{}.txt".format(prog, v, prog, ctype)
    fault_matrix = "../input/{}_{}/fault_matrix.pickle".format(prog, v)

    ptimes, stimes, apfds = [], [], []
    for run in range(repeats):
        print(" Run", run)
        if name == "ART-D":
            stime, ptime, prioritization = competitors.art_d(fin)
        elif name == "ART-F":
            stime, ptime, prioritization = competitors.art_f(fin)
        else:
            print("Wrong input.")
            print(usage)
            exit()
        apfd = metric.apfd(prioritization, fault_matrix)
        apfds.append(apfd)
        stimes.append(stime)
        ptimes.append(ptime)
        print("  Progress: 100%  ")

    return (prioritization, stime + ptime, sum(apfds[run]) / len(apfds[run]))
    print("")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def prioritize(prog_v, entity, algname, repeats):

    repeats = int(repeats)
    algnames = {"ART-D", "ART-F"}
    prog_vs = {"closure_v0", "lang_v0", "math_v0", "chart_v0", "time_v0"}
    entities = {"function", "branch", "line"}

    if prog_v not in prog_vs:
        print("<dataset> input incorrect.")
        print(usage)
        exit()
    elif entity not in entities:
        print("<entity> input incorrect.")
        print(usage)
        exit()
    elif algname not in algnames:
        print("<algorithm> input incorrect.")
        print(usage)
        exit()
    elif repeats <= 0:
        print("<repetitions> input incorrect.")
        print(usage)
        exit()

    prog, v = prog_v.split("_")

    return wboxPrioritization(algname, prog, v, entity, repeats)

if __name__ == "__main__":

    prog_v = "chart_v0"
    entity = "line"
    algname = "ART-D"
    repeats = "1"

    order, time, apfd = prioritize(prog_v, entity, algname, repeats)
    print("order: ", order)
    print("time: ", time)
    print("apfd:", apfd)
