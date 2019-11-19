import argparse

def gettestargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filter", help="Use file as filter")
    args = parser.parse_args()
    return args
