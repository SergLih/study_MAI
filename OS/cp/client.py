import zmq
import sys
from battleship import *
from api import *
from enum import IntEnum
from time import sleep


class Player(IntEnum):
    ME = 0
    OPPONENT = 1


class GameClient:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # milliseconds
        self.socket.connect('tcp://localhost:8888')

        self.fields = [Field(), Field()]  # 0 - mine, 1 - opponent

        self.ID = 0
        self.gameID = 0
        self.gameOver = False

        # настройки
        self._requestInterval = 1
        self._numberOfReceiveAttempts = 30


    def _send(self, *args):
        self.socket.send("{} {} {} {}".format(self.gameID, self.ID, args[0].value,
                        (' '.join(map(str, args[1:])) if len(args) > 1 else "")).encode("ascii"))

    def _receive(self):
        for i in range(self._numberOfReceiveAttempts):
            try:
                return list(map(int, self.socket.recv().split()))
            except zmq.error.Again:
                self._connError()
                print('Attempt #{}/{}'.format(i+1, self._numberOfReceiveAttempts))
                sleep(self._requestInterval)
        self.Stop("Maximum number of attempts exceeded.")


    def _connError(self):
        print("Connection with the server has been lost. ", end='')


    def Start(self):
        self.fields = [Field(), Field()]  # 0 - mine, 1 - opponent

        self.ID = 0
        self.gameOver = False

        self.gameID = int(input("Input game ID to join existing game or 0 to create new one: "))
        self._send(Command.START)
        data = self._receive()

        if data[0] == Result.OK:
            self.gameID, self.ID = data[1:]
            print("Your opponent can join the game with this ID:", self.gameID)
            self.SetShips()
            self.MakeMoves()
        else:
            self.Stop(data[1])


    def Stop(self, error=None):
        if error:
            print(error)
        print("Game over.")
        self.gameOver = True
        self.socket.close(linger=0)
        self.context.term()
        sys.exit(1 if error else 0)

    def processChangedCells(self, fieldNo, changedCells):
        cells = changedCells[1:]
        for i in range(0, changedCells[0]*3, 3):
            x, y, state = cells[i:i + 3]
            self.fields[fieldNo][x, y] = Cell(CellState(state)) # в клиентской версии можно не делать ID корабля

    def _printOneField(self):
        f0 = self.fields[Player.ME].AsCharArray(showAliveShips=True)
        print ('   ABCDEFGHIJ\n' + \
            '\n'.join(["{:2} {}".format(i + 1, f0[i]) for i in range(10)]) + '\n')


    def _printTwoFields(self):
        f0 = self.fields[Player.ME].AsCharArray(showAliveShips=True)
        f1 = self.fields[Player.OPPONENT].AsCharArray(showAliveShips=False)
        print('   ABCDEFGHIJ\tABCDEFGHIJ\n' + \
               '\n'.join(["{:2} {}\t{}".format(i+1, f0[i], f1[i]) for i in range(10)]) + '\n')


    def SetShips(self):
        for shipLength in range(4, 0, -1):
            for shipCount in range(5 - shipLength):
                if shipLength == 1:
                    print("Input coordinate of 1-cell ship #{}: ".format(shipCount+1))
                    x1, y1 = self.InputCoordinate(input())
                    x2, y2 = x1, y1
                else:
                    print("Input coordinates of {}-cells ship #{}: ".format(shipLength, shipCount+1))
                    coords = input().split()
                    x1, y1 = self.InputCoordinate(coords[0])
                    x2, y2 = self.InputCoordinate(coords[1])

                self._send(Command.SET_SHIP, x1, y1, x2, y2)
                data = self._receive()

                if data[0] == Result.OK:
                     self.processChangedCells(Player.ME, data[1:])
                     self._printOneField()
                else:
                    self.Stop(data[1])

        # TODO: ждать пока можно сделать ход
        print("Waiting while opponent is setting their ships...", end='')
        while True:
            self._send(Command.WAIT_FOR_MOVE)
            data = self._receive()

            if data[0] == Result.OK:
                print(flush=True)
                break
            elif data[0] == Result.ERROR and data[1] == ErrorType.E_CANNOT_MOVE_WAITING_OPPONENT:
                print(flush=True)
                self.WaitForMove()
                break
            elif data[0] == Result.ERROR and data[1] == ErrorType.E_INVALID_GAME_ID_ALREADY_STARTED:
                self.Stop("ERROR: Wrong game ID. Maybe your opponent cancelled the game.")
            sleep(self._requestInterval)
            print('.', end='', flush=True)



    def MakeMoves(self):
        while not self.gameOver:
            # 1. Сделать ход
            self._printTwoFields()
            x, y = self.InputCoordinate(input("Input coordinate of your shot: "))

            self._send(Command.SHOT, x, y)
            data = self._receive()


            # 2. Результат хода:
            #    Продолжение игры -- ожидание хода противника (0)
            #    Дополнительный ход                           (1)
            #    Победа                                       (2)
            if data[0] == Result.OK:
                self.processChangedCells(Player.OPPONENT, data[2:])
                self._printTwoFields()
                if data[1] in [MoveResult.HIT, MoveResult.DESTROYED]:
                    print("You can make extra move now!")
                    continue
                elif data[1] == MoveResult.MOVE_REPEAT:
                    print('You have already shot to this cell, please repeat: ')
                    continue
                elif data[1] == MoveResult.WIN:
                    print('You win!')
                    self.AskForRestart()
                self.WaitForMove()

            else:
                self.Stop(1)


    def WaitForMove(self):
        print('\nWaiting while opponent is making their move...', end='')
        while not self.gameOver:
            self._send(Command.WAIT_FOR_MOVE)
            data = self._receive()
            if data[0] == Result.OK:
                if len(data) > 2:
                    self.ProcessOpponentMove(data[1:])
                break
            elif data[0] == Result.ERROR and data[1] == ErrorType.E_INVALID_GAME_ID_ALREADY_STARTED:
                print("ERROR: Wrong game ID. Maybe your opponent cancelled the game.")
                self.gameOver = True
                break
            else:
                if len(data) > 3:
                    self.ProcessOpponentMove(data[2:])
            sleep(self._requestInterval)
            print('.', end='', flush=True)
        print(flush=True)

    def ProcessOpponentMove(self, data):
        print("Opponent made their move:")
        self.processChangedCells(Player.ME, data[1:])
        self._printTwoFields()
        if data[0] == MoveResult.HIT:
            print('Your ship has been damaged! Waiting for extra move of the opponent...')
        elif data[0] == MoveResult.DESTROYED:
            print('Your ship has been destroyed! Waiting for extra move of the opponent...')
        elif data[0] == MoveResult.WIN:
            print('Your ship has been destroyed and you lose!')
            self.AskForRestart()
        else:
            print("Your opponent shot and missed the target")


    def AskForRestart(self):
        self.gameOver = True
        if input('Do you want to play another game? (y/n) ') == 'y':
            self.Start()
        else:
            sys.exit(0)

    def InputCoordinate(self, str_coord):
        # TODO: проверка что буквы и цифры правильные
        x = int(str_coord[1:]) - 1
        y = ord(str_coord[:1]) - ord('A')
        return x, y


if __name__ == '__main__':
    cl = GameClient()
    cl.Start()


    #
    #
    # def get(self, key):
    #     self.socket.send(pickle.dumps(('get', key, None)))
    #     return pickle.loads(self.socket.recv())
    #
    # def set(self, key, data):
    #     self.socket.send(pickle.dumps(('set', key, data)))
    #     return self.socket.recv() == b'ok'
