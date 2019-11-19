import argparse

def gettrainargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--qmatrix", help="Use file as Q matrix")
    parser.add_argument("-v", "--vmatrix", help="Use file as V matrix")
    args = parser.parse_args()
    return args
