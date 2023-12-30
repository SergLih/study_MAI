//
// Created by sergey on 5/20/21.
//

//#ifndef EXPAT_TEST_INDEX_H
//#define EXPAT_TEST_INDEX_H

#include <iostream>
//#include <expat.h>
#include "json.hpp"
#include "hashtable.hpp"
#include "exprtree.hpp"
#include "vector.hpp"

using json = nlohmann::json;



//void serialize_dsize_t(const TItem&, ofstream &);
//void deserialize_dsize_t(TItem&, ifstream &);

class TIndex {

private:
    THashTable ht;
    TVector<TString> authors;
    TVector<TString> titles;
    TVector<TString> dois;

//    static void reset_char_data_buffer ();
//    static void char_data(void *userData, const XML_Char *s, int len);
//    static void process_char_data_buffer (void);
//    static void XMLCALL startElement(void *userData, const XML_Char *name, const XML_Char **atts);
//    static void XMLCALL endElement(void *userData, const XML_Char *name);
    void deserialize(std::ifstream & file);
    void serialize(std::ofstream & file);

    TVector<dsize_t> getDocs(TString word);
    TVector<dsize_t> evaluateTree_rec(Node* start);

public:
    double get_average_word_length();
    dsize_t get_docs_count();

    void build(string jsonl_filename);
    void load(string ind_filename);
    void save(string ind_filename);

    void search(string query);

    string to_string();
    //for tests
    string ht_tostring();


};
//
//#endif //EXPAT_TEST_INDEX_H
