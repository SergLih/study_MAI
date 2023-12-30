//
// Created by sergey on 5/20/21.
//
#include <iomanip>
#include <regex>
#include "index.h"

//void TIndex::reset_char_data_buffer () {
//    ss_content.str("");
//    read_char_data = false;
//}
//
//// pastes parts of the node together
//void TIndex::char_data (void *userData, const XML_Char *s, int len) {
//    if(read_char_data)
//        ss_content << string(s, len);
//}

//void TIndex::process_char_data_buffer (void) {
//
//    string text = ss_content.str();
//    // cout << "Doc: " << id_doc << "\nContent:\t" << ss_content.str() << "\n";
//    std::transform(text.begin(), text.end(), text.begin(),
//                   [](unsigned char c){ return std::tolower(c); });
//    regex reg(R"([a-z][a-z\d]+)");
//    regex reg_gene( "[acgt]{6,}" );  // 5'-acgtgtg-3'
//    sregex_token_iterator start(text.begin(), text.end(), reg), end;
//    for (auto it=start;  it!=end; ++it) {
//#ifdef DEBUG
//        cerr << *it << " " << id_doc << endl;
//#endif
//        if(it->length() > 255) {
//            cerr << "Warning: word '" << *it << "' is too long, skipping...\n";
//            continue;
//        }
//        if(regex_search(it->str(), reg_gene))
//        {
//            cerr << "Warning: word '" << *it << "' looks like a long DNA sequence, skipping...\n";
//            continue;
//        }
//        ht.insert(it->str(), id_doc);
//    }
//    //cout << text;
//}
//
//void XMLCALL
//TIndex::startElement(void *userData, const XML_Char *name, const XML_Char **atts) {
//    string tag_name = string(name);
//    if(tag_name == "Article")
//        id_doc++;
//    read_char_data = (tag_name == "Abstract" || tag_name == "Body");
//}
//
//void XMLCALL
//TIndex::endElement(void *userData, const XML_Char *name) {
//    string tag_name = string(name);
//    if(tag_name == "Body") {
//        process_char_data_buffer();
//        reset_char_data_buffer();
//    }
//}
//
double TIndex::get_average_word_length() {
    return ht.get_sum_length_keys()/double(ht.get_terms_count());
}

dsize_t TIndex::get_docs_count() {
    return titles.size();
}



void TIndex::build(string jsonl_filename) {
    ifstream fin_jsonl(jsonl_filename, ios::binary);
    string line;
    dsize_t id_doc = 0;
    while (std::getline(fin_jsonl, line)) {
        auto data = json::parse(line);
        authors.push(TString(data["Authors"].is_null() ? "" : data["Authors"] )); // индексы в векторе авторов соответствуют id документам
        titles.push(TString(data["Title"].is_null() ? "" : data["Title"]));
        dois.push(TString(data["Doi"].is_null() ? "" : data["Doi"]));

        string text = string(data["Abstract"].is_null() ? "" : data["Abstract"]) +
                      " " + string(data["Body"].is_null() ? "" : data["Body"]);

        std::transform(text.begin(), text.end(), text.begin(),
                       [](unsigned char c) { return std::tolower(c); });

        regex reg(R"([a-z][a-z\d]+)");
        regex reg_gene("[acgt]{6,}");  // 5'-acgtgtg-3'
        sregex_token_iterator start(text.begin(), text.end(), reg), end;
        for (auto it = start; it != end; ++it) {
#ifdef DEBUG
            cerr << *it << " " << id_doc << endl;
#endif
            if (it->length() > 255) {
                cerr << "Warning: word '" << *it << "' is too long, skipping...\n";
                continue;
            }
            else if (regex_search(it->str(), reg_gene)) {
                cerr << "Warning: word '" << *it << "' looks like a long DNA sequence, skipping...\n";
                continue;
            }
            ht.insert(TString(it->str()), id_doc);
        }
        id_doc++;
    }
    ht.printAll();
    ht.printSizeCapacity();
}

string TIndex::ht_tostring() {
    ht.printAll();
    return ht.to_string();
}

string TIndex::to_string() {
    stringstream ss;
    ss << ht_tostring() << endl;
    ss << "Count terms: " << ht.get_terms_count() << endl;
    ss << "Docs count: " << get_docs_count() << endl;
    ss << "Average word length: " << get_average_word_length() << endl;
    return ss.str();
}

void TIndex::deserialize(std::ifstream &file) {
    ht.deserialize(file);
    authors.deserialize(file);
    titles.deserialize(file);
    dois.deserialize(file);
}

void TIndex::save(string ind_filename) {
    ofstream fout_ht(ind_filename, ios::binary);
    serialize(fout_ht);
    fout_ht.close();
}

void TIndex::load(string ind_filename) {
    ifstream bin_file(ind_filename, ios::binary);
    deserialize(bin_file);
    bin_file.close();
}

void TIndex::serialize(std::ofstream &fout_ht) {
    ht.serialize(fout_ht);
    authors.serialize(fout_ht);
    titles.serialize(fout_ht);
    dois.serialize(fout_ht);
}

//Using a recursive function, the value of the expression is Calculated
TVector<dsize_t> TIndex::evaluateTree_rec(Node* start){

    //TString x,y,z;
    //int order;
    if ( start->info == "OR" || start->info == "AND") {
        TVector<dsize_t> res_left  = evaluateTree_rec(start->left);
        TVector<dsize_t> res_right = evaluateTree_rec(start->right);
        if (start->info == "OR") {
            return res_left._union(res_right);
        }
        else if (start->info == "AND") {
            return res_left.intersect(res_right);
        }
    }
    else {
        return getDocs(start->info);
    }
}


void TIndex::search(string query) {
    TExprTree t(query);
    TVector<dsize_t> res = evaluateTree_rec(t.treeNodes.top());
    for (int i = 0; i < res.size(); ++i) {
        cout << res.get(i) << "\t" << titles.get(i) << endl;
    }
}

TVector<dsize_t> TIndex::getDocs(TString word) {
    size_t h = ht.getHash(word);
    return ht.values[h];
}

////    uint64_t xml_file_sz;
////    try {
////        xml_file_sz = fs::file_size(jsonl_filename); // attempt to get _size of a directory
////    } catch (fs::filesystem_error &e) {
////        std::cout << e.what() << '\n';
////        exit(1);
////    }
//    //cerr << "Loading file " << jsonl_filename << " of _size  " << xml_file_sz / 1024 << "KB\n";
//
//
//
//
//
////    char buf[BUFSIZ];
////    XML_Parser parser = XML_ParserCreate(NULL);
////    (void) argc;
////    (void) argv;
////    bool done = false;
////
////    XML_SetElementHandler(parser, startElement, endElement);
////    XML_SetCharacterDataHandler(parser, char_data);
////
////    unsigned int start_time = clock();
////    reset_char_data_buffer();
////    uint64_t read = 0;
////    uint64_t on_disk_sz, in_memory_sz;
////    do {
////        fin_jsonl.read(buf, sizeof(buf));
////        size_t len = fin_jsonl.gcount();
////        read += len;
////        if(read % (10000 * 1024) < sizeof(buf)){
////            on_disk_sz = ht.get_size_on_disk();
////            in_memory_sz = ht.get_size_in_memory();
////        }
////        cerr << "\rReading... " << fixed << setprecision(2) << read * 100.0 / xml_file_sz << "%\t"
////             << read / 1024 << " / " << xml_file_sz / 1024 << " KB"
////             << "\tPredicted HT _size on disk / in memory: "
////             << on_disk_sz / 1024 << " / " << in_memory_sz / 1024 <<  " KB";
////        //        size_t len = fread(buf, 1, sizeof(buf), stdin);
////        done = len < sizeof(buf);
////        if (XML_Parse(parser, buf, (int) len, done) == XML_STATUS_ERROR) {
////            fprintf(stderr, "%" XML_FMT_STR " at line %" XML_FMT_INT_MOD "u\n",
////                    XML_ErrorString(XML_GetErrorCode(parser)),
////                    XML_GetCurrentLineNumber(parser));
////            XML_ParserFree(parser);
////            return;
////        }
////    } while (!done);
////    XML_ParserFree(parser);
////    fin_jsonl.close();
////    unsigned int end_time = clock();
////    cout << "\n\nRead _data from xml file: " << ((float)(end_time - start_time))/CLOCKS_PER_SEC << " seconds\n";
////
////    unsigned int start_time2 = clock();
////    ofstream fout_ht(ht_filename, ios::binary);
////    ht.serialize(fout_ht);
////    fout_ht.close();
////    unsigned int end_time2 = clock();
////    cout << "\nSerialize: " << ((float)(end_time2 - start_time2))/CLOCKS_PER_SEC << " seconds\n";
////    cout << "\n";
////    ht.printAll();
////    ht.printSizeCapacity();
//}
//
//void TIndex::load(string ind_filename) {
//
//}
//
//void TIndex::save(string ind_filename) {
//
//}

