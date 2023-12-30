#include "TFile.h"
#include "TLCS_finder.h"

int main(int argc, char *argv[]) {
	if (argc < 3) {
		cout << "Usage: " << argv[0] << "[<options>] <file1> <file2>\n";
		return 1;
	}
	TFile file1, file2;
	bool ignoreNewLines = false;

	if (argc == 4 && argv[1][0] == '-') {
		file1 = TFile(argv[2]);
		file2 = TFile(argv[3]);
		string options(argv[1]);
		int nOptions = 0;
		if(options.find('i') != string::npos) {
			file1.TransformToLower();
			file2.TransformToLower();
			//cout << "i\n";
			nOptions++;
		}
		if (options.find('w') != string::npos) {
			file1.TransformRemoveAllSpaces();
			file2.TransformRemoveAllSpaces();
			//cout << "w\n";
			ignoreNewLines = true;
			nOptions++;
		}
		if (options.find('B') != string::npos) {
			file1.TransformRemoveBlankLines();
			file2.TransformRemoveBlankLines();
			//cout << "B\n";
			nOptions++;
		}
		if (options.find('Z') != string::npos) {
			file1.TransformRemoveTrailingSpaces();
			file2.TransformRemoveTrailingSpaces();
			//cout << "Z\n";
			ignoreNewLines = true;
			nOptions++;
		}
		if(nOptions+1 < options.size()) {
			cout << "Error: " << argv[1] << " contains wrong option keys!" << endl;
			return 2;
		}
	}
	else if (argc == 3) {
		file1 = TFile(argv[1]);
		file2 = TFile(argv[2]);
	}
	if (file1.status == Status::empty || file2.status == Status::empty) {
		return 1;
	}

	TLCS_Finder finder(file1, file2, ignoreNewLines);
	finder.PrintEditScript();

	return 0;
}
