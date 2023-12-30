//
// Created by sergey on 4/25/21.
//

#ifndef LABS_VECTOR_HPP
#define LABS_VECTOR_HPP

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include "tstring.h"

using namespace std;

const static int VECTOR_DEFAULT_CAPACITY = 2;
const static double VECTOR_EXTENSION_FACTOR = 1.5;




//Только растущий с конца вектор для значений size_t
template <typename TItem>
class TVector {

private:
    TItem * _arr;
    uint32_t _size;
    uint32_t _capacity;

    void expand()
    {
        //int old_capacity = _capacity;
        _capacity = int(VECTOR_EXTENSION_FACTOR * _capacity);
        TItem* temp = new TItem[_capacity];
        for (size_t i = 0; i < _size; ++i) {
            temp[i] = _arr[i];
        }
        //memcpy(temp, _arr, sizeof(TItem)*old_capacity);
        delete[] _arr;
        _arr = temp;
    }

public:

    TVector() {
        _arr = new TItem[VECTOR_DEFAULT_CAPACITY];
        _capacity = VECTOR_DEFAULT_CAPACITY;
        _size = 0;
    }

    TVector & operator=(const TVector & other)
    {
        _capacity = other._capacity;
        _size = other._size;
        if (_arr != nullptr)
            delete[] _arr;
        _arr  = new TItem[_capacity];
        memcpy(_arr, other._arr, _capacity * sizeof(TItem));
        return *this;
    }

    bool operator==(const TVector<TItem> &other) const;

    ~TVector(){
        delete[] _arr;
    }

    void push(const TItem& data) {
        _arr[_size++] = data;
        if (_size == _capacity)
            expand();
    }

    TItem top() {
        return _arr[_size - 1];
    }

    TItem get(int index) {
        if (index < _size)
            return _arr[index];
        else
            cerr << "Index out of range: " << index << " / " << _size << "\n";
    }

    bool empty() { return _size == 0; }
    uint32_t size() { return _size; }
    uint32_t capacity() { return _capacity; }

    //for tests only
    string to_string(){
        stringstream ss;
        for (uint32_t i = 0; i < _size; i++) {
            ss << _arr[i] << "\t";
        }
        return ss.str();
    }

    size_t get_size_on_disk() {
        return sizeof(TItem)*(_size) + sizeof(uint32_t);
    }

    size_t get_size_in_memory() {
        return sizeof(TItem)*(_capacity) + 2*sizeof(uint32_t);
    }

    void serialize(std::ofstream & file) const {
        file.write(reinterpret_cast<const char *>(&(_size)), sizeof(uint32_t));
        file.write(reinterpret_cast<const char *>(_arr), sizeof(TItem) * _size);
    }

    void deserialize(ifstream &file) {
        if (file) {
            file.read(reinterpret_cast<char *>(&(_size)), sizeof(uint32_t));
            _capacity = _size;
            if (_arr)
                delete [] _arr;
            _arr  = new TItem[_size];
            file.read(reinterpret_cast<char *>(_arr), sizeof(TItem) * _size);
        } else {
            cerr << "error: only " << file.gcount() << " could be read";
        }
    }

    TVector<TItem> intersect(TVector<TItem> &v2) {
        TVector res;
        uint32_t i = 0, j = 0;
        uint32_t size1 = this->_size;
        uint32_t size2 = v2._size;
        while ((i < size1) && (j < size2)) {
            if (this->get(i) == v2.get(j)) {
                res.push(this->get(i));
                i++;
                j++;
            } else if (this->get(i) < v2.get(j))
                i++;
            else j++;
        }
        return res;
    }

    TVector<TItem> _union(TVector<TItem> &v2) {
        TVector res;
        uint32_t i = 0, j = 0;
        uint32_t size1 = this->_size;
        uint32_t size2 = v2._size;
        while ((i < size1) && (j < size2)) {
            if (this->get(i) == v2.get(j)) {
                res.push(this->get(i));
                i++;
                j++;
            } else if (this->get(i) < v2.get(j)) {
                res.push(this->get(i));
                i++;
            } else {
                res.push(v2.get(j));
                j++;
            }
        }
        while (i < size1)
            res.push(this->get(i++));
        while (j < size2)
            res.push(v2.get(j++));

        return res;
    }

};

template<typename TItem>
bool TVector<TItem>::operator==(const TVector<TItem> &other) const {
    if(_size != other._size)
        return false;

    for (size_t i = 0; i < _size; ++i) {
        if(!(_arr[i] == other._arr[i]))
            return false;
    }
    return true;
}

//template<typename TItem>
//inline ostream &operator<<(ostream &os, const TVector<TItem> &v) {
//    for (uint32_t i = 0; i < v._size; i++) {
//        os << v._arr[i] << " ";
//    }
//    return os;
//}

template<>
inline size_t TVector<TString>::get_size_on_disk(){
    size_t total = sizeof(uint32_t);
    for (size_t i = 0; i < this->_size; ++i) {
        total += (_arr[i].size() + 1);
    }
    return total;
}

template<>
inline size_t TVector<TString>::get_size_in_memory() {
    size_t total = 2*sizeof(uint32_t);
    for (size_t i = 0; i < this->_capacity; ++i) {
        total += (_arr[i].size() + 1);
    }
    return total;
}

template<>
inline void TVector<TString>::serialize(std::ofstream & file) const
{
    file.write(reinterpret_cast<const char *>(&(_size)), sizeof(uint32_t));
    for (size_t i = 0; i < _size; i++) {
        this->_arr[i].serialize(file);
    }
}

template<>
inline void TVector<TString>::deserialize(ifstream &file)
{
    if (file) {
        file.read(reinterpret_cast<char *>(&(_size)), sizeof(uint32_t));
        _capacity = _size;
        if (_arr)
            delete [] _arr;
        _arr  = new TString[_size];
        for (size_t i = 0; i < _size; i++) {
            _arr[i].deserialize(file);
        }
    } else {
        cerr << "error: only " << file.gcount() << " could be read";
    }
}

#endif //LABS_VECTOR_HPP
