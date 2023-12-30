#ifndef TFILE
#define TFILE

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string> 
#include <cstdlib>

using namespace std;

enum class Status {
	empty, read
};

class TFile {
public:
	TFile();
	TFile(char * filename);
	TFile(const TFile &file);

	void TransformToLower();              //i
	void TransformRemoveAllSpaces();      //w
	void TransformRemoveBlankLines();     //B
	void TransformRemoveTrailingSpaces(); //Z
	Status status;

private:
	
	bool noNewLineAtEOF;

	vector<string> lines;

	friend class TLCS_Finder;

};

#endif // !TFILE