#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <zmq.hpp>

#include <unistd.h>
#include <string>
#include <sys/mman.h>
#include <sys/file.h>
#include <sys/types.h>

using namespace std;

const int PORT_ADDRESS = 5556;
const int FIELD_SIZE = 10;

enum class Player {
    ME,
    OPPONENT
};

enum class Command {
    START,
    WAIT_FOR_MOVE,
    SET_SHIP,
    SHOT
};

enum class Result {
    OK,
    ERROR
};

enum class GameState {
    NOT_STARTED,    
    SETTING_SHIPS,  
    IN_PROGRESS,    
    FINISHED       
};

enum class ShipState {
    ALIVE,
    DEAD
};

enum class CellState {
    EMPTY,
    PROTECTED,
    MISS,
    ALIVE,
    DAMAGED,
    DESTROYED
};


enum class ErrorType {
    E_SHIP_SIZE_INCORRECT,
    E_ALL_SHIPS_OF_THIS_SIZE_SET,
    E_CANNOT_SET_SHIP_ON_OCC_CELL,
    E_INVALID_GAME_ID_DOES_NOT_EXIST,
    E_INVALID_GAME_ID_ALREADY_STARTED,
    E_INVALID_COORDINATES,
    E_CANNOT_START_NOT_ALL_SHIPS_ARE_SET,
    E_CANNOT_MOVE_WAITING_OPPONENT,
    E_CANNOT_MOVE_GAME_OVER,
    E_CANNOT_MOVE_REPEAT
};

enum class MoveResult {
    MISS,
    HIT,
    DESTROYED,
    WIN
};

class GameClient {
    public:
        CellState fields[2][FIELD_SIZE][FIELD_SIZE];
        int ID = 0;
        int gameID = 0;
        bool gameOver = false;

        double _requestIntervalFactor = 2;
        int _numberOfReceiveAttempts = 30;

        zmq::context_t* context;
        zmq::socket_t*  socket;
	
	
    GameClient();
    GameClient(int port);
    ~GameClient();
	
    vector<int> _receive();
    void  _send(vector<int> args);
    void InitFields();
    void Start();
    void Stop(const char* error = nullptr);
    void processChangedCells(int fieldNo, vector<int> &changedCells);
    char CellStateChar(CellState s);
    void _printFields(bool printOpponent=true);
    int  ShipSize(int x1, int y1, int x2, int y2);
    void SetShips();
    void MakeMoves();
    void WaitForMove();
    void ProcessOpponentMoves(vector<int> data);
    void AskForRestart();
    bool InputCoordinate(int &x, int &y);
};

GameClient::GameClient(): GameClient(PORT_ADDRESS) {}

GameClient::GameClient(int port) {
    this->context = new zmq::context_t(1);
    this->socket  = new zmq::socket_t(*(this->context), ZMQ_REQ);
    int timeout = 1000;
    this->socket->setsockopt(ZMQ_RCVTIMEO, &timeout, sizeof (int));
    string addr = string("tcp://127.0.0.1:") + to_string(port);
    this->socket->connect(addr);
}		

GameClient::~GameClient() {
    delete this->context;
    delete this->socket;
}


void GameClient::_send(vector<int> args) {
    std::ostringstream oss;
    oss << this->gameID << " " << this->ID;
    for(int i = 0; i<args.size(); i++) {
        oss << " " << args[i];
    }

    zmq::message_t msg (oss.str().length());
    memcpy ((void *) msg.data (), (void*)oss.str().c_str(), oss.str().length());
    if(! this->socket->send(msg))
        this->Stop("Cannot send data to the socket.\n");
}


vector<int> GameClient::_receive() {
    for (int i = 0; i < this->_numberOfReceiveAttempts; i++) {
        int n_trial = 1;
        zmq::message_t msg;
        if(!this->socket->recv(&msg)) {
            cout << "Connection with the server has been lost. Attempt #" << i + 1 << "/" << this->_numberOfReceiveAttempts << endl;
            cout.flush();
            int interval = min(int(pow(this->_requestIntervalFactor, n_trial++)), 30);
            sleep(interval);
        } else {
            vector<int> data;
            string s(static_cast<char*>(msg.data()), msg.size());
            string elem;
            istringstream iss(s);
            while (getline(iss, elem, ' ')) {
               data.push_back(stoi(elem));
            }
            return data;
        }
    }
    this->Stop("Maximum number of attempts exceeded.");
}

int GameClient::ShipSize(int x1, int y1, int x2, int y2) {
    int d1 = abs(x2 - x1) + 1;
    int d2 = abs(y2 - y1) + 1;
    if (d1 == 1 || d2 == 1) {
        return max(d1, d2); //размер корабля
    } else
        return 0;           //некорректный размер
}

void GameClient::InitFields() {
    for(int p = 0; p<=1; p++)
        for(int i =0; i<FIELD_SIZE; i++)
            for(int j =0; j<FIELD_SIZE; j++)
                this->fields[p][i][j] = CellState::EMPTY;
}

void GameClient::Start() {
    
    this->InitFields();
    this->ID = 0;
    this->gameOver = false;

    cout << "Input game ID to join existing game or 0 to create new one: ";
    cin >> this->gameID;
    vector<int> args {(int)Command::START};
    this->_send(args);
    vector<int> data = this->_receive(); 

    if (static_cast<Result>(data[0]) == Result::OK) {
        if (this->gameID == 0)
            cout << "Your opponent can join the game with this ID: " << data[1] << endl;
        this->gameID = data[1];
        this->ID = data[2];

        this->SetShips();
        this->MakeMoves();
    } else {
        this->Stop(to_string(data[1]).c_str());
    }
}

void GameClient::Stop(const char* error) {
    if (!error) {
        cerr << error;
    }
    cout << "Game over\n";
    this->gameOver = true;
    this->socket->close();
    if (!error)
        exit(EXIT_FAILURE);
    else
        exit(EXIT_SUCCESS);
}

void GameClient::processChangedCells(int fieldNo, vector<int> &changedCells)
{
    //размер вектора в нулевом элементе вектора
    for (int i = 1; i < changedCells[0]*3 + 1; i+=3) {
        int x     = changedCells[i];
        int y     = changedCells[i+1];
        CellState state = static_cast<CellState>(changedCells[i+2]);
        this->fields[fieldNo][x][y] = state;
    }
    changedCells.erase(changedCells.begin(), changedCells.begin() + changedCells[0]*3 + 1);
}

char GameClient::CellStateChar(CellState s) {
    switch(s) {
        case CellState::EMPTY:     return '.';
        case CellState::PROTECTED: return '.';
        case CellState::MISS:      return 'o';
        case CellState::ALIVE:     return '@';
        case CellState::DAMAGED:    return 'X';
        case CellState::DESTROYED:  return '#';
        default: return ' ';
    }
}

void GameClient::_printFields(bool printOpponent) {
    cout << "   ABCDEFGHIJ";
    if(printOpponent)
        cout << "\tABCDEFGHIJ\n";
    else
        cout << "\n";
        
    for (int i = 0; i < FIELD_SIZE; i++) {
        cout << setw(2) << i+1 << " ";
        for(int j = 0; j < FIELD_SIZE; j++)  
            cout << CellStateChar(fields[(int)Player::ME][i][j]);

        if(printOpponent) {
            cout << "\t";
            for(int j = 0; j < FIELD_SIZE; j++)  
            cout << CellStateChar(fields[(int)Player::OPPONENT][i][j]);
            cout << "\n";
	
        } else
            cout << "\n";
    }
}

void GameClient::SetShips( ) {
    int x1, y1, x2, y2;
    vector<int> data;
    for (int shipLength = 4; shipLength > 0; shipLength--) {
        for (int shipCount = 0; shipCount < 5 - shipLength; shipCount++) {
            if (shipLength == 1) {
                cout << "Input coordinate of 1-cell ship #" << shipCount+1 << ": ";
                while(!this->InputCoordinate(x1, y1))
                    cerr << "Incorrect coordinate. Input coordinate like A2, D3, etc. Try again: ";
                x2 = x1;
                y2 = y1;
            } else {
                int length;
                bool error;
                cout << "Input coordinates of " << shipLength << "-cells ship #" << shipCount+1 << ": ";
                do {
                    error = false;
                    error |= !this->InputCoordinate(x1, y1);
                    error &= !this->InputCoordinate(x2, y2); 
                    
                    if (error) {
                        cerr << "Incorrect coordinates. Input two coordinates like A2, D3, etc. Try again: ";
                    } else {                    
                        length = ShipSize(x1, y1, x2, y2);
                        
                        if (length != shipLength) {
                            cerr << "Incorrect ship size! Try again: ";
                        }
                    } 
                    
                } while(error || length != shipLength);
            }
            vector<int> args {(int)Command::SET_SHIP, x1, y1, x2, y2};
            this->_send(args);
            data = this->_receive();

            if (static_cast<Result>(data[0]) == Result::OK) {
                data.erase(data.begin());
                this->processChangedCells((int)Player::ME, data);
                this->_printFields(false);
            } else
                this->Stop(to_string(data[1]).c_str());
        }
    }
	
    cout << "Waiting while opponent is setting their ships..."; 
    vector<int> args {(int)Command::WAIT_FOR_MOVE};
    int n_trial = 1;
    while (true) {
        this->_send(args);
        data = this->_receive();

        if (static_cast<Result>(data[0]) == Result::OK) {
            cout << endl;
            break;
        }

        if (static_cast<Result>(data[0]) == Result::ERROR 
	               && static_cast<ErrorType>(data[1]) == ErrorType::E_CANNOT_MOVE_WAITING_OPPONENT) {
            cout << endl;
            this->WaitForMove();
            break;
        }

        if (static_cast<Result>(data[0]) == Result::ERROR 
                   && static_cast<ErrorType>(data[1]) == ErrorType::E_INVALID_GAME_ID_ALREADY_STARTED) {
            this->Stop("ERROR: Wrong game ID. Maybe your opponent cancelled the game.");
        }
        cout.flush();
        int interval = min(int(pow(this->_requestIntervalFactor, n_trial++)), 30);
        sleep(interval);
        cout << ".";
    }
}

void GameClient::MakeMoves() {
    int x, y;
    vector<int> data;
    while (!this->gameOver) {
        // 1. Сделать ход
        this->_printFields();
        cout << "Input coordinate of your shot: ";
        while(!this->InputCoordinate(x, y))
            cerr << "Incorrect coordinate. Input coordinate like A2, D3, etc. Try again: ";
        vector<int> args {(int)Command::SHOT, x, y};
        this->_send(args);
        data = this->_receive();

        // 2. Результат хода:
        //    Продолжение игры -- ожидание хода противника (0)
        //    Дополнительный ход                           (1)
        //    Победа                                       (2)

        if (static_cast<Result>(data[0]) == Result::OK) {
            MoveResult res = static_cast<MoveResult>(data[1]);
            data.erase(data.begin(), data.begin()+2);
            processChangedCells((int)Player::OPPONENT, data);
            this->_printFields();
            if(res == MoveResult::HIT || res == MoveResult::DESTROYED) {
                cout << "You can make extra move now!\n";
                continue;	
            } else if (res == MoveResult::WIN) {
                cout << "You win!\n";
                this->AskForRestart();
            }
            this->WaitForMove();
        } else if(static_cast<Result>(data[0]) == Result::ERROR && 
                  static_cast<ErrorType>(data[1]) == ErrorType::E_CANNOT_MOVE_REPEAT) {
            cout << "You have already shot to this cell, please repeat.\n";
            continue;    
        }
        else
            this->Stop("1");
    }
}

void GameClient::WaitForMove() {
    vector<int> data;
    cout << "Waiting while opponent is making their move..." << endl;
    int n_trial = 1;
    while (!this->gameOver) {
        vector<int> args {(int)Command::WAIT_FOR_MOVE};
        this->_send(args);
        data = this->_receive();
        if (static_cast<Result>(data[0]) == Result::OK) {
            if (data.size() > 2) {
                data.erase(data.begin());
                this->ProcessOpponentMoves(data);
            }
            break;
        } else if (static_cast<Result>(data[0]) == Result::ERROR) {
            if(static_cast<ErrorType>(data[1]) == ErrorType::E_INVALID_GAME_ID_ALREADY_STARTED) {
                cout << "ERROR: Wrong game ID. Maybe your opponent cancelled the game." << endl;
                this->gameOver = true;
                break;
            } else if (data.size() > 3) {
                data.erase(data.begin(), data.begin()+2); // Ошибка и тип ошибки
                this->ProcessOpponentMoves(data);
            }
        }
        cout.flush();
        int interval = min(int(pow(this->_requestIntervalFactor, n_trial++)), 30);
        sleep(interval);
        cout << ".";
    }
    cout << endl;
}

void GameClient::ProcessOpponentMoves(vector<int> data) {
    int numberOfMoves = data[0];
    data.erase(data.begin());

    if(numberOfMoves == 1)
        cout << "Opponent made their move:" << endl;
    else
        cout << "Opponent made several moves:" << endl;
        
    for(int i = 0; i<numberOfMoves; i++) {
        MoveResult res = static_cast<MoveResult>(data[0]);
        data.erase(data.begin());

        this->processChangedCells((int)Player::ME, data);
        this->_printFields();
        if (res == MoveResult::HIT) {
            cout << "Your ship has been damaged! Waiting for extra move of the opponent..." << endl;
        } else if (res == MoveResult::DESTROYED) {
            cout << "Your ship has been destroyed! Waiting for extra move of the opponent..." << endl;
        } else if (res == MoveResult::WIN) {
            cout << "Your ship has been destroyed and you lose!" << endl;
            this->AskForRestart();
        } else {
            cout << "Your opponent shot and missed the target" << endl;
        }
    }
}

void GameClient::AskForRestart() {
    this->gameOver = true;
    cout << "Do you want to play another game? (y/n) " << endl;
    string answer;
    cin >> answer;
    if (answer == "y")
        this->Start();
    else
        exit(EXIT_SUCCESS);
}

bool GameClient::InputCoordinate(int &x, int &y) {
    string s;
    bool error;
    error = false;
    cin >> s;
    if (s.length() > 3) {
        error = true;
    } else {
        try {
            x = stoi(s.substr(1)) - 1;
        }
        catch(...) {
            error = true;
        }
        y = s[0] - 'A';
        if(y < 0 || y > 10 || x < 0 || x > 10)
            error = true;
    }
    return (!error);
}

int main(int argc, char *argv[]) {
    int port = 5556;
    if(argc == 2) {
        bool error = false;
        string s(argv[1]);
        size_t pos = s.length();
        try {
            port = stoi(s, &pos);
            error = port < 1 || port > 65536;
        } catch(...) {
            error = true;
        }
        if(pos != s.length()) {
            error = true;
        }        
        if(error) {
            cerr << "Incorrect port number" << endl;
            return 1;
        }
    }
    GameClient cl(port);
    cl.Start();
    return 0;
}
