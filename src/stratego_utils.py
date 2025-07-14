import csv


SPY = 0
SCOUT = 1
MINER = 2
SERGEANT = 3
LIEUTENANT = 4
CAPTAIN = 5
MAJOR = 6
COLONEL = 7
GENERAL = 8
MARSHAL = 9
BOMB = 10
FLAG = 11

START = 12

NUM_PIECE_TYPES = 12
NUM_SETUP_SQUARES = 40
NUM_SETUP_ROWS = 4
NUM_SETUP_COLUMNS = 10

PIECE_COUNTS = {
    SPY: 1, SCOUT: 8, MINER: 5, SERGEANT: 4, LIEUTENANT: 4,
    CAPTAIN: 4, MAJOR: 3, COLONEL: 2, GENERAL: 1, MARSHAL: 1,
    BOMB: 6, FLAG: 1
}

PIECE_TO_STR = {
    SPY: "1", SCOUT: "2", MINER: "3", SERGEANT: "4", LIEUTENANT: "5",
    CAPTAIN: "6", MAJOR: "7", COLONEL: "8", GENERAL: "9", MARSHAL: "M",
    BOMB: "B", FLAG: "F"
}


def read_setups(path, unique=True):
    setups = []
    with open(path, "r") as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            setups.append(tuple(int(piece) for piece in row))
    if not unique:
        return setups
    unique_setups = []
    seen = set()
    for setup in setups:
        if setup not in seen:
          unique_setups.append(setup)
          seen.add(setup)
    return unique_setups
