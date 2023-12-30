//
// Created by sergey on 5/14/21.
//
#ifndef EXPAT_TEST_TSTRING_H
#define EXPAT_TEST_TSTRING_H


#include <string>
#include <cstring>
#include <iostream>
#include <fstream>

using namespace std;

class TString {

private:
    uint8_t _size;
    char * _data;

public:
    TString();
    TString(const TString& other);
    explicit TString(string s);
    explicit TString(const char *);
    virtual ~TString();

    uint8_t size();
    bool empty();
    size_t hash() const;

    friend ostream& operator<<(ostream& os, const TString& s);
    bool operator==(const TString &other) const;
    bool operator==(const string &other) const;
    bool operator==(const char* &other) const;
    bool operator<(const TString &other) const;
    TString & operator=(const TString & other);

    void serialize(std::ofstream & file) const;

    void deserialize(ifstream & file);

};

#endif //EXPAT_TEST_TSTRING_H
