//                 lexer.h 2018
#ifndef LEXER_H
#define LEXER_H
#include "baselexer.h"
//********************************************
//*        Developed by  xxx		     *
//*             (c)  2018                    *
//********************************************
class tLexer:public tBaseLexer{
public:
// персональный код разработчика
 std::string Authentication()const{
                     return "SSL"
                +std::string("2018");}
//конструктор
 tLexer():tBaseLexer(){
//создать автоматы:

//  ноль Azero
    addstr  (Azero,0,"+-", 2);
    addstr  (Azero,0,"0",  1);
    addstr  (Azero,2,"0",  1);
  Azero.final(1);

//________________________________________

// число
  addrange(Adec,  0, '1','9', 2);
  addstr  (Adec,  0,   "+",   1);
  addstr  (Adec,  0,   "-",   1);
  addstr  (Adec,  0,   "0",   8);
  addstr  (Adec,  1,   "0",   8);
  addstr  (Adec,  8,   ".",   3);
  addstr  (Adec,  8,   "e",   5);
  addstr  (Adec,  8,   "E",   5);
  addrange(Adec,  1, '1','9', 2);
  addrange(Adec,  2, '0','9', 2);
  addstr  (Adec,  2,   ".",   3);
  addstr  (Adec,  2,   "e",   5);
  addstr  (Adec,  2,   "E",   5);
  addrange(Adec,  3, '0','9', 4);
  addrange(Adec,  4, '0','9', 4);
  addstr  (Adec,  4,   "e",   5);
  addstr  (Adec,  4,   "E",   5);
  addstr  (Adec,  5,   "+",   6);
  addstr  (Adec,  5,   "-",   6);
  addrange(Adec,  6, '0','9', 7);
  addrange(Adec,  7, '0','9', 7);
 Adec.final(2);
 Adec.final(4);
 Adec.final(7);
//________________________________________

// идентификатор
    addrange(Aid, 0, 'a', 'z', 1);
    addrange(Aid, 0, 'A', 'Z', 1);
    addstr  (Aid, 0,   "!",    2);
    addrange(Aid, 1, 'a', 'z', 1);
    addstr  (Aid, 1,   "-",    1);
    addrange(Aid, 1, 'A', 'Z', 1);
    addrange(Aid, 1, '0', '9', 1);
    addstr  (Aid, 1,   "!",    2);
  Aid.final(1);
  Aid.final(2);
//________________________________________

// идентификатор предиката
    addrange(Aidq, 0, 'a', 'z', 2);
    addrange(Aidq, 0, 'A', 'Z', 2);   
    addstr  (Aidq, 0,   "?",    1);
    addstr  (Aidq, 0,   "-",    3);
    addstr  (Aidq, 1,   "?",    1);
    addstr  (Aidq, 3,   "-",    3);
    addstr  (Aidq, 3,   "?",    1);
    addrange(Aidq, 3, 'a', 'z', 2);
    addrange(Aidq, 3, 'A', 'Z', 2);
    addrange(Aidq, 3, '0', '9', 2);
    addrange(Aidq, 1, 'a', 'z', 2);
    addrange(Aidq, 1, 'A', 'Z', 2);
    addrange(Aidq, 1, '0', '9', 2);
    addrange(Aidq, 2, '0', '9', 2);
	addrange(Aidq, 2, 'a', 'z', 2);
    addrange(Aidq, 2, 'A', 'Z', 2);
    addstr  (Aidq, 2,   "?",    1);
    addstr  (Aidq, 2,   "-",    3);
  Aidq.final(1);
}
};
#endif

