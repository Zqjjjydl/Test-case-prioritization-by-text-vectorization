from collections import defaultdict
from pickle import load


def apfd(prioritization, fault_matrix):
    """INPUT:
    (list)prioritization: list of prioritization of test cases
    (str)fault_matrix: path of fault_matrix (pickle file)

    OUTPUT:
    (float)APFD = 1 - (sum_{i=1}^{m} t_i / n*m) + (1 / 2n)
    n = number of test cases
    m = number of faults detected
    t_i = position of first test case revealing fault i in the prioritization
    Average Percentage of Faults Detected
    """

    # key=version, val=[faulty_tcs]
    faults_dict = getFaultDetected(fault_matrix)
    apfds = []
    for v in range(1, len(faults_dict)+1):
        faulty_tcs = set(faults_dict[v])
        numerator = 0.0  # numerator of APFD
        position = 1
        m = 0.0
        for tc_ID in prioritization:
            if tc_ID in faulty_tcs:
                numerator, m = position, 1.0
                break
            position += 1

        n = len(prioritization)
        apfd = 1.0 - (numerator / (n * m)) + (1.0 / (2 * n)) if m > 0 else 0.0
        apfds.append(apfd)

    return apfds



def getFaultDetected(fault_matrix):
    """INPUT:
    (str)fault_matrix: path of fault_matrix (pickle file)

    OUTPUT:
    (dict)faults_dict: key=tcID, val=[detected faults]
    """
    faults_dict = defaultdict(list)

    with open(fault_matrix, "rb") as picklefile:
        pickledict = load(picklefile)
    for key in pickledict.keys():
        faults_dict[int(key)] = pickledict[key]

    return faults_dict
