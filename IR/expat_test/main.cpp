#include <iostream>
#include <string>
//#include <string_view>
#include <regex>
#include <cstdio>

#include <cstring>
#include <typeinfo>
#include <sstream>
#include <fstream>
#include <experimental/filesystem>
#include <iomanip>
#include <ctime>

namespace fs = std::experimental::filesystem;
//#include "exprtree.hpp"
#include "index.h"
//#include <shunting-yard.h>

#include "vector.hpp"

//using namespace std;
using std::string;
using std::cin;
using std::cout;

int main(int argc, char *argv[]) {

    TIndex ind;
    if (string(argv[1]) == "build") {
        if (argc < 4){
            cerr << "Usage: ./expat_test build <file.xml> <hashtable.dat>";
            exit(1);
        }
        string jsonl_filename = string(argv[2]);
        string ht_filename = string(argv[3]);
        ind.build(jsonl_filename);
        ind.save(ht_filename);
        cout << ind.to_string();
    } else if (string(argv[1]) == "load") {
        //cout << "Hiii :)))))))))\n";
        // Usage: ./expat_test load <hashtable.dat>
        if (argc < 3) {
            cerr << "Usage: ./expat_test load <hashtable.dat>";
            exit(1);
        }
        string filename = string(argv[2]);
        ind.load(filename);
        cout << ind.to_string();

        string s;
        do{
            std::getline(cin, s);
            ind.search(s);

        } while(s != "");
    }


//    string s;
//    //cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
////    std::getline(cin, s);
//s = "lol AND kek";
//    TExprTree res(s);
//    res.evaluateTree();
//    cout << "end\n";
    return 0;
}
