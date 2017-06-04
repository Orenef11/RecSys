import datetime
import random
import sys

from scipy.sparse import coo_matrix
from scipy.io import mmwrite

def get_timestamp():
    return random.randrange(datetime.datetime(2017, 1, 1, 0, 0, 0).timestamp(), datetime.datetime(2017, 12, 31, 23, 59, 59).timestamp())

def main():
    M = coo_matrix([[0, get_timestamp(), get_timestamp(), 0], 
                    [get_timestamp(), 0, 0, get_timestamp()], 
                    [get_timestamp(), get_timestamp(), 0, 0],
                    [0, 0, get_timestamp(), get_timestamp()],
                    [0, 0, get_timestamp(), 0]])
    mmwrite(sys.argv[1], M)


if __name__ == '__main__':
    main()