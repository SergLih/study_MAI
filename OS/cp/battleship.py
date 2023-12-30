from collections import namedtuple
from enum import IntEnum

class Command(IntEnum):
    START         = 0
    WAIT_FOR_MOVE = 1
    SET_SHIP      = 2
    SHOT          = 3

class Result(IntEnum):
    OK    = 0
    ERROR = 1

class GameState(IntEnum):
    NOT_STARTED    = 0
    SETTING_SHIPS  = 1
    IN_PROGRESS    = 2
    FINISHED       = 3


class ShipState(IntEnum):
    ALIVE = 0
    DEAD  = 1

class CellState(IntEnum):
    EMPTY = 0
    PROTECTED = 1
    MISS  = 2
    ALIVE = 3
    DAMAGED = 4
    DESTROYED = 5

class ErrorType(IntEnum):
    E_SHIP_SIZE_INCORRECT                = 0
    E_ALL_SHIPS_OF_THIS_SIZE_SET         = 1
    E_CANNOT_SET_SHIP_ON_OCC_CELL        = 2
    E_INVALID_GAME_ID_DOES_NOT_EXIST     = 3
    E_INVALID_GAME_ID_ALREADY_STARTED    = 4
    E_INVALID_COORDINATES                = 5
    E_CANNOT_START_NOT_ALL_SHIPS_ARE_SET = 6
    E_CANNOT_MOVE_WAITING_OPPONENT       = 7
    E_CANNOT_MOVE_GAME_OVER              = 8
    E_CANNOT_MOVE_REPEAT                 = 9


class MoveResult(IntEnum):
    MISS        = 0
    HIT         = 1
    DESTROYED   = 2
    WIN         = 3


class FieldUtils:
    def CorrectShip(coords):
        dims = FieldUtils.ShipDimensions(coords)
        return min(dims) == 1 and 1 <= max(dims) <= 4


    def ShipDimensions(coords):
        (x1, y1), (x2, y2) = coords
        return abs(x2 - x1) + 1, abs(y2 - y1) + 1

class Cell:
    def __init__(self, state, shipID = None):
        self.State = state
        self.ShipID = shipID


class Ship:
    nextID = 0

    def __init__(self, coords):

        self.ID = Ship.nextID
        Ship.nextID += 1

        (x1, y1), (x2, y2) = coords
        size = max(FieldUtils.ShipDimensions(coords))
        self.size = size
        self.aliveCellsCount = size
        self.state = ShipState.ALIVE

        if x1 == x2: # horizontal
            self.cells = [(x1, y) for y in range(min(y1, y2), max(y1, y2)+1)]
        else:        # vertical
            self.cells = [(x, y1) for x in range(min(x1, x2), max(x1, x2)+1)]

    def GetProtectedCells(self):
        protected_cells = []
        for x in range(max(0, self.cells[0][0] - 1),
                       min(9, self.cells[-1][0] + 1) + 1):
            for y in range(max(0, self.cells[0][1] - 1),
                       min(9, self.cells[-1][1] + 1) + 1):
                if (x, y) not in self.cells:
                    protected_cells.append((x, y))
        return protected_cells


class Field:
    Cell = namedtuple('Cell', 'State ShipID')
    def __init__(self):
        self._f = [[Cell(CellState.EMPTY)
                    for j in range(10)] for i in range(10)]


    def StateToSymbol(self, state, showAliveShips=True):
        charmap = ['.', '.', 'o', '@', 'X', '#']
        if not showAliveShips:
            charmap[3] = '.'

        d = dict(zip(list(CellState), charmap))
        return d[state]

    def AsCharArray(self, showAliveShips=True):
        return [''.join([self.StateToSymbol(self._f[i][j].State, showAliveShips)
                         for j in range(10)]) for i in range(10)]

    def __getitem__(self, coords):
        x, y = coords
        return self._f[x][y]

    def __setitem__(self, coords, val):
        x, y = coords
        self._f[x][y] = val



class Game:
    def __init__(self):
        self.state = GameState.NOT_STARTED
        self.fields = [Field(), Field()]
        self.unsetShipSizes = [{1: 4, 2: 3, 3: 2, 4: 1}, {1: 4, 2: 3, 3: 2, 4: 1}]
        self.ships = dict()
        self.aliveShipsCount = [10, 10]
        self.currentPlayer = 0

    def Start(self):
        self.state = GameState.IN_PROGRESS

    def _changePlayer(self):
        self.currentPlayer = 1 - self.currentPlayer

    def PrintFieldsForPlayer(self, playerNo):
        if playerNo == 0:
            f0 = self.fields[0].AsCharArray(showAliveShips=True)
            f1 = self.fields[1].AsCharArray(showAliveShips=False)
        else:
            f0 = self.fields[1].AsCharArray(showAliveShips=True)
            f1 = self.fields[0].AsCharArray(showAliveShips=False)

        return '   ABCDEFGHIJ\tABCDEFGHIJ\n' + \
               '\n'.join(["{:2} {}\t{}".format(i+1, f0[i], f1[i]) for i in range(10)]) + '\n'


    def SetShip(self, playerNo, *coords):
        if not FieldUtils.CorrectShip(coords):
            raise ValueError(ErrorType.E_SHIP_SIZE_INCORRECT)

        ship = Ship(coords)
        if self.unsetShipSizes[playerNo][ship.size] == 0:
            raise ValueError(ErrorType.E_ALL_SHIPS_OF_THIS_SIZE_SET)

        if not all(self.fields[playerNo][ship_cell_coords].State == CellState.EMPTY
                   for ship_cell_coords in ship.cells):
            raise ValueError(ErrorType.E_CANNOT_SET_SHIP_ON_OCC_CELL)

        self.unsetShipSizes[playerNo][ship.size] -= 1
        changes = dict()
        for ship_cell_coords in ship.cells:
            self.fields[playerNo][ship_cell_coords] = Cell(CellState.ALIVE, ship.ID)
            changes[ship_cell_coords] = CellState.ALIVE

        for pr_coords in ship.GetProtectedCells():
            self.fields[playerNo][pr_coords] = Cell(CellState.PROTECTED, None)
        self.ships[ship.ID] = ship
        if self.GameIsReadyToStart():
            self.state = GameState.IN_PROGRESS

        return changes

    def Shot(self, playerNo, coords):
        changes = dict()
        result = None
        if self.fields[1 - playerNo][coords].State in [CellState.EMPTY, CellState.PROTECTED]:
            self.fields[1 - playerNo][coords].State = CellState.MISS
            changes[coords] = CellState.MISS
            result = MoveResult.MISS
            self._changePlayer()
        elif self.fields[1 - playerNo][coords].State == CellState.ALIVE:
            if (self.ships[self.fields[1 - playerNo][coords].ShipID]).aliveCellsCount > 1:
                self.fields[1 - playerNo][coords].State = CellState.DAMAGED
                changes[coords] = CellState.DAMAGED
                result = MoveResult.HIT
            else:
                for ship_cell_coords in self.ships[self.fields[1 - playerNo][coords].ShipID].cells:
                    self.fields[1 - playerNo][ship_cell_coords].State = CellState.DESTROYED
                    changes[ship_cell_coords] = CellState.DESTROYED
                self.ships[self.fields[1 - playerNo][coords].ShipID].state = ShipState.DEAD
                self.aliveShipsCount[1 - playerNo] -= 1
                if self.aliveShipsCount[1 - playerNo] == 0:
                    result = MoveResult.WIN
                else:
                    result = MoveResult.DESTROYED

            self.ships[self.fields[1 - playerNo][coords].ShipID].aliveCellsCount -= 1
        else:
            raise ValueError(ErrorType.E_CANNOT_MOVE_REPEAT)
            
        return result, changes

    def GameIsReadyToStart(self):
        return sum(self.unsetShipSizes[0].values()) + sum(self.unsetShipSizes[1].values()) == 0
