#include "TFile.h"

TFile::TFile() {
	status = Status::empty;
}

TFile::TFile(char * filename) {
	ifstream infile(filename);

	if (!infile.is_open()) {
		cerr << "diff: " << filename << " : No such file or directory\n";
		status = Status::empty;
		return;
	}

	string line;
	while (infile.good()) {
		getline(infile, line);		//\n в конце
		lines.push_back(line/* + "\n"*/);
	}
	size_t last_idx = lines.size() - 1;
	//size_t last_len = lines[last_idx].size();
	if (lines[last_idx] == /*"\n"*/"") {
		lines.pop_back();
		noNewLineAtEOF = false;
	} else {
		//lines[last_idx] = lines[last_idx].substr(0, last_len - 1);
		noNewLineAtEOF = true;
	}

	infile.close();
	status = Status::read;
}

TFile::TFile(const TFile & other) {
	this->lines = other.lines;
	this->status = other.status;
	this->noNewLineAtEOF = other.noNewLineAtEOF;
}

void TFile::TransformToLower() {
	for (string &line : lines) {
		transform(line.begin(), line.end(), line.begin(), ::tolower);
	}
}

void TFile::TransformRemoveAllSpaces() {
	for (string &line : lines) {
		line.erase(remove(line.begin(), line.end(), ' '), line.end());
		line.erase(remove(line.begin(), line.end(), '\t'), line.end());
	}
}

void TFile::TransformRemoveBlankLines() {
	lines.erase(remove(lines.begin(), lines.end(), /*isspace*/"" /*"\n"*/), lines.end());
}

void TFile::TransformRemoveTrailingSpaces() {
	for (string &line : lines) {
		//string::iterator end = (line[line.size() - 1] == '\n') ? line.end() - 1 : line.end();
		line.erase(find_if_not(line.rbegin(), line.rend(), ::isspace).base(), line.end() /*end*/);
	}
}
