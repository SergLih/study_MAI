//
// Created by sergey on 5/14/21.
//

#include "tstring.h"

TString::TString(string s) {
    _size = s.length();          //error if len > 255
    if(_size > 255) {
        cerr << "Warning: too long string \"" << s << "\", only first 255 characters will be saved\n";
        _size = 255;
    }
    _data = new char[_size+1];
    memcpy(_data, s.c_str(), _size+1);
}

TString::TString(const char * s) {
    _size = strlen(s);
    if(_size > 255) {
        cerr << "Warning: too long string \"" << s << "\", only first 255 characters will be saved\n";
        _size = 255;
    }
    _data = new char[_size+1];
    strcpy(_data, s);
}


TString::~TString() {
    if (this->_data)
        delete this->_data;
}

size_t TString::hash() const {
    return std::hash<string>{}(string(_data));
}

bool TString::empty() {
    return _size == 0;
}

TString::TString() {
    _data = nullptr;
    _size = 0;
}

TString &TString::operator=(const TString &other) {
    if(this != &other)
    {
        _size = other._size;
        if(_data != nullptr)
            delete[] _data;
        _data  = new char[_size+1];
        strcpy(_data, other._data);
    }
    return *this;
}

ostream& operator<<(ostream& os, const TString& s)
{
    os << reinterpret_cast<char *>(s._data);
    return os;
}

uint8_t TString::size() {
    return _size;
}

void TString::serialize(ofstream &file) const {
    file.write(reinterpret_cast<const char *>(&(_size)), sizeof(uint8_t));
    file.write(reinterpret_cast<char *>(_data), sizeof(char)*_size);
//    for (uint8_t i = 0; i < _size; ++i) {
//        file.write(reinterpret_cast<const char *>(&_data[i]), sizeof(char));
//    }
}


bool TString::operator==(const TString &other) const {
    return strncmp(this->_data, other._data, this->_size) == 0;
}

bool TString::operator==(const string &other) const {
    return strncmp(this->_data, other.c_str(), this->_size) == 0;
}

bool TString::operator==(const char *&other) const {
    return strncmp(this->_data, other, this->_size) == 0;
}



void TString::deserialize(ifstream &file) {
    if (file) {
        file.read(reinterpret_cast<char *>(&(this->_size)), sizeof(this->_size));
        //if (this->_data)
        //    delete [] this->_data;
        this->_data  = new char[this->_size+1];
        file.read(this->_data, this->_size);
        this->_data[this->_size] = '\0';
    } else {
        cerr << "error: only " << file.gcount() << " could be read";
    }
}

bool TString::operator<(const TString &other) const {
    return strcmp(_data, other._data) < 0;
}

TString::TString(const TString &other) {
        _size = other._size;
        _data  = new char[_size+1];
        strcpy(_data, other._data);
}


