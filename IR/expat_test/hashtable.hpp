//
// Created by sergey on 4/25/21.
//

#ifndef LABS_HASHTABLE_HPP
#define LABS_HASHTABLE_HPP

//#define HT_DEBUG

#include <iostream>
#include <functional>
#include <string_view>
#include <iomanip>
#include "vector.hpp"
#include "tstring.h"

using namespace std;


typedef uint32_t dsize_t; //для корпуса размером до 2^32 ~ 4 * 10^9 документов
//typedef uint16_t dsize_t; //для корпуса размером до 2^16 = 65536 документов

const static int DEFAULT_CAPACITY = 8;
const static double EXTENSION_FACTOR = 1.5;

//key: string, value: TVector<size_t>
class THashTable{
    friend class TIndex;
private:
    TString *keys;
    TVector<dsize_t>* values;
    uint32_t size;
    uint32_t capacity;


    bool expandIfNeeded() {
        if(size > 0.75 * capacity) {
            #ifdef HT_DEBUG
                cerr << " EXPAND! \n";
            #endif
            size_t old_capacity = capacity;
            capacity = int(VECTOR_EXTENSION_FACTOR * capacity);
            auto* new_keys = new TString[capacity];                //word
            auto* new_values = new TVector<dsize_t>[capacity];     //doc numbers
            for (size_t h = 0; h < old_capacity; ++h) {
                if(!keys[h].empty()){
                    size_t new_h = keys[h].hash() % capacity;
                    while(!(new_keys[new_h].empty()))
                        new_h = (new_h + 1) % capacity;
                    new_keys[new_h] = keys[h];
                    new_values[new_h] = values[h];  //operator=()
                }
            }
            delete[] keys;
            delete[] values;
            keys = new_keys;
            values = new_values;
            return true;
        }
        return false;
    }

public:
    // Constructor to create a hash table with 'n' indices:
    THashTable(){
        size = 0;
        capacity = DEFAULT_CAPACITY;
        keys = new TString[capacity];       //word
        values = new TVector<dsize_t>[capacity];     //doc numbers
        //values = new TVector<TString>[capacity];
    }

    ~THashTable(){
        delete[] keys;
        delete[] values;
    }

    void insert(TString s, /*TString author*/size_t docnumber) {

        size_t h = getHash(s);
        if (keys[h].empty())     //еще такой ключ не вставляли
        {
            size++;
            if(expandIfNeeded())
                h = getHash(s);
        } else {        // уже вставляли
            // проверяем не встречался ли этот токен в этом же документе только что
            if (!values[h].empty() && values[h].top() == docnumber)
                return;
        }

        keys[h] = s;
        values[h].push(docnumber);

    }

    string to_string(){
        vector<TString> tmp_keys(size);
        for (uint32_t h = 0, j=0; h < capacity; h++) {
            if(!keys[h].empty()){
                tmp_keys[j++] = keys[h];
            }
        }
        sort(tmp_keys.begin(), tmp_keys.end());

        stringstream ss;
        for (uint32_t i = 0; i < size; i++) {
            TString k = tmp_keys[i];
            ss << k << ":\t";
            size_t h = getHash(k);
            ss << values[h].to_string() << endl;
        }
        return ss.str();
    }

    void printAll(){
        for (size_t h = 0; h < capacity; ++h) {
            if(!keys[h].empty()){
                cout << "[" << h << "]\t" << keys[h] << ":\t";
                cout << values[h].to_string() << endl;
            }
        }
    }

    dsize_t get_sum_length_keys() {
        dsize_t sum = 0;
        for (size_t h = 0; h < capacity; ++h) {
            if (!keys[h].empty()) {
                sum += keys[h].size();
            }
        }
        return sum;
    }

    dsize_t get_terms_count() {
        return size;
    }

    void printSizeCapacity() {
        cout << "Size: " << size << "\nCapacity: " << capacity << endl;
    }

    size_t get_size_on_disk() {
        uint64_t res = 4; //_size
        for (size_t h = 0; h < capacity; ++h) {
            if (!keys[h].empty()) {
                res += keys[h].size() + 1 + values[h].get_size_on_disk();
            }
        }
        return res;
    }

    size_t get_size_in_memory() {
        uint64_t res = sizeof(size_t)*capacity*2;
        for (size_t h = 0; h < capacity; ++h) {
            res += keys[h].size() + values[h].get_size_in_memory();
        }
        return res;
    }

    void serialize(std::ofstream & file) {
        file.write(reinterpret_cast<const char *>(&(size)), sizeof(uint32_t));
        for (size_t h = 0; h < capacity; ++h) {
            if (!keys[h].empty()) {
                keys[h].serialize(file);
                values[h].serialize(file);
            }
        }

    }

    void deserialize(std::ifstream & file) {
        if (file) {
            file.read(reinterpret_cast<char *>(&(this->size)), sizeof(uint32_t));
            this->capacity = int(this->size*1.2);
            if (this->values)
                delete [] this->values;
            if (this->keys)
                delete [] this->keys;
            keys = new TString[capacity];
            values = new TVector<dsize_t>[capacity];
            //values = new TVector<TString>[capacity];
            TString tmp_key;



            for (size_t h = 0; h < size; ++h) {
                cerr << "\rReading... " << h << " / " << size
                     << fixed << setprecision(2) << "\t(" << h * 100.0 / size << "%)";
                tmp_key.deserialize(file);
                size_t hash = getHash(tmp_key);
                while(!(keys[hash].empty()))
                    hash = (hash + 1) % capacity;
                keys[hash] = tmp_key;
                values[hash].deserialize(file);
            }
        } else {
            cerr << "error: only " << file.gcount() << " could be read";
        }
    }

    // Hash function to calculate hash for a value:
    size_t getHash(const TString &word){
        #ifdef HT_DEBUG
            cerr << "\tH\t" << word << " : " << std::hash<std::string>{}(word) % capacity << "\n";
        #endif
        size_t h  =  word.hash() % capacity;
        // мы ищем свободное место (возможно с разрешением коллизий), либо натыкаемся на это слово
        while(!(keys[h].empty() || keys[h] == word))
            h = (h + 1) % capacity;
        return h;
    }
};



#endif //LABS_HASHTABLE_HPP
