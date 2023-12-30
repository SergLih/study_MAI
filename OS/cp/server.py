from battleship import *

import sys
import zmq
import logging
import signal

class GameServer:
    def __init__(self, port = 5556):
        self.port = port;
        self.games = dict()
        self.nextGameID = 1
        self.nextPlayerID = 1
        self.playersInGame = dict()     # ключ - ID игры, значения - ID игроков {1: [1, 2], 2: [3, 4], 5: [8, 11]}
        self.moveResults = dict()

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind('tcp://127.0.0.1:{}'.format(port))

        self.running = True
        self.deleteFinishedGames = True


        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d.%m.%y %H:%M:%S')
        self.logger = logging.getLogger('tipper')
        self.logger.setLevel(logging.DEBUG)


    def _sendOK(self, *args):
        self.logger.debug('  Sent  OK: ' + ' '.join(map(str, args)))
        if len(args) > 0:
            self.socket.send("{} {}".format(Result.OK.value,  ' '.join(map(str, args))).encode("ascii"))
        else:
            self.socket.send(str(Result.OK.value).encode("ascii"))

    def _sendError(self, errorType, *args):
        self.logger.debug('  Sent ERR: ' + str(errorType) + " " + ' '.join(map(str, args)))
        self.socket.send("{} {} {}".format(
            Result.ERROR.value, errorType.value, ' '.join(map(str, args))).encode("ascii"))

    def _serializeMoveResults(self, moveResults):
        res = str(len(moveResults)) + ' '
        for moveResult in moveResults:                
            res += str(moveResult[0].value) + ' ' + self._serializeCellsChanges(moveResult[1]) + ' '
        return res

    def _serializeCellsChanges(self, cellsChanges):
        return str(len(cellsChanges)) + " " + \
               ' '.join("{} {} {}".format(*coords, state.value) for coords, state in cellsChanges.items())

    def Run(self):
        self.logger.info('Server started on port {}'.format(self.port))
        while self.running:
            try:
                data = self.socket.recv().decode("ascii")
                data = list(map(int, data.split()))
                gameID, playerID, command = data[:3]

                self.logger.info('Game {}, player {}'.format(gameID, playerID))
                self.logger.info('  Received: {} {}'.format(Command(command).name, 
                ((' '.join(map(str, data[3:]))) if len(data) > 3 else "")))

                if command == Command.START:   # старт игры
                    if gameID == 0:            # создаем новую игру
                        gameID = self.nextGameID
                        playerID = self.nextPlayerID
                        self.games[gameID] = Game()
                        self.playersInGame[gameID] = [playerID]

                        self.nextPlayerID += 1
                        self.nextGameID += 1

                        self._sendOK(gameID, playerID)

                    else:              # подключаемся к существующей игре
                        if gameID not in self.games:
                            self._sendError(ErrorType.E_INVALID_GAME_ID_DOES_NOT_EXIST)
                        elif self.games[gameID].state != GameState.NOT_STARTED:
                            self._sendError(ErrorType.E_INVALID_GAME_ID_ALREADY_STARTED)
                        else:
                            playerID = self.nextPlayerID
                            self.playersInGame[gameID].append(playerID)
                            
                            self.nextPlayerID += 1
                            self._sendOK(gameID, playerID)
                            self.games[gameID].state = GameState.SETTING_SHIPS
                            

                elif command == Command.SET_SHIP: # установка кораблей
                    if gameID not in self.games:  # такой игры нет
                        self._sendError(ErrorType.E_INVALID_GAME_ID_DOES_NOT_EXIST)
                    elif self.games[gameID].state != GameState.SETTING_SHIPS: # игра уже идет или закончена
                        self._sendError(ErrorType.E_INVALID_GAME_ID_ALREADY_STARTED)
                    else:
                        try:
                            x1, y1, x2, y2 = data[3:]
                            try:
                                changedCells = self.games[gameID].SetShip(self.playersInGame[gameID].index(playerID), (x1, y1), (x2, y2))
                                self._sendOK(self._serializeCellsChanges(changedCells))
                            except ValueError as err:
                                self._sendError(err.args[0])
                        except:
                            self._sendError(ErrorType.E_INVALID_COORDINATES)

                elif command == Command.WAIT_FOR_MOVE:
                    if gameID not in self.games:
                        self._sendError(ErrorType.E_INVALID_GAME_ID_DOES_NOT_EXIST)
                    elif self.games[gameID].state == GameState.IN_PROGRESS:
                        # если очередь того, кто отпр. запрос
                        if self.playersInGame[gameID].index(playerID) == self.games[gameID].currentPlayer:
                            if playerID in self.moveResults: #если сообщение о рез-те хода хранилось для этого игрока
                                self._sendOK(self._serializeMoveResults(self.moveResults[playerID]))
                                del self.moveResults[playerID]
                            else:
                                self._sendOK()
                        else: # ожидание хода другого игрока
                            if playerID in self.moveResults:
                                self._sendError(ErrorType.E_CANNOT_MOVE_WAITING_OPPONENT, 
                                                self._serializeMoveResults(self.moveResults[playerID]))
                                
                                if self.moveResults[playerID][-1][0] == MoveResult.WIN:
                                    self.games[gameID].state = GameState.FINISHED
                                    if self.deleteFinishedGames:
                                        del self.games[gameID]
                                del self.moveResults[playerID]
                            else:
                                self._sendError(ErrorType.E_CANNOT_MOVE_WAITING_OPPONENT)

                    elif self.games[gameID].state in [GameState.SETTING_SHIPS, GameState.NOT_STARTED]:
                        self._sendError(ErrorType.E_CANNOT_START_NOT_ALL_SHIPS_ARE_SET)
                    else:
                        self._sendError(ErrorType.E_INVALID_GAME_ID_ALREADY_STARTED)

                elif command == Command.SHOT:
                    if gameID not in self.games:
                        self._sendError(ErrorType.E_INVALID_GAME_ID_DOES_NOT_EXIST)
                    elif self.games[gameID].state == GameState.IN_PROGRESS:
                        # если очередь того, кто отпр. запрос
                        if self.playersInGame[gameID].index(playerID) == self.games[gameID].currentPlayer:
                            try:
                                x, y = data[3:]
                                try:
                                    # выясним айпи оппонента до выстрела, потому что после выстрела может поменяться игрок 
                                    opponentID = self.playersInGame[gameID][1 - self.games[gameID].currentPlayer]
                                    
                                    shotResult, shotCellChanges = \
                                        self.games[gameID].Shot(self.playersInGame[gameID].index(playerID), (x, y))
                                    
                                    if opponentID in self.moveResults:
                                        self.moveResults[opponentID] += [(shotResult, shotCellChanges)]
                                    else:
                                        self.moveResults[opponentID] = [(shotResult, shotCellChanges)]
                                    #отправляем ответ стрелявшему изменение. Стреляющий получает информацию сразу.
                                    self._sendOK(shotResult.value, self._serializeCellsChanges(shotCellChanges)) 

                                except ValueError as err:
                                    self._sendError(err.args[0])
                            except:
                                self._sendError(ErrorType.E_INVALID_COORDINATES)
                        else:
                            self._sendError(ErrorType.E_CANNOT_MOVE_WAITING_OPPONENT)
                    else:
                        self._sendError(ErrorType.E_CANNOT_MOVE_GAME_OVER)

            except KeyboardInterrupt:
                  self.Stop()
                  break
            except Exception as e:
                  print('Error')
                  self.logger.error('ERROR: ', e)
                  self.Stop()
                  break

    def Stop(self):
        if self.running:
            self.running = False
            self.logger.info('Server stopped')
            self.socket.close(linger=0)
            self.context.term()


if __name__ == '__main__':
    port = 5556
    if len(sys.argv) == 2:
        error = False
        try:
            port = int(sys.argv[1]);
            error = not (0 < port < 65536) 
        except:
            error = True
        
        if error:
            sys.exit("Incorrect port number!");
            
    serv = GameServer(port)
    serv.Run()
    serv.Stop()
