#ifndef TLCS_FINDER
#define TLCS_FINDER

#define DEBUG

#include "TFile.h"
#include <cstring>
#include <vector>

typedef string TBlock;
typedef int TLineNumber;

enum class Step {
	Up, Left, UpLeft
};

enum class Action {
	Change, Delete, Append
};

struct TDiffAction {
	Action action;
	TLineNumber start1;
	TLineNumber end1;
	TLineNumber start2;
	TLineNumber end2;

	TDiffAction(Action action, TLineNumber s1, TLineNumber e1, TLineNumber s2, TLineNumber e2) {
		this->action = action;
		start1 = s1;
		end1 = e1;
		start2 = s2;
		end2 = e2;
	}

	string ToString() {
		string s = "";
		if (start1 == end1)
			s += to_string(start1);
		else
			s += to_string(start1) + "," + to_string(end1);

		switch (action) {
		case Action::Change: s += "c"; break;
		case Action::Delete: s += "d"; break;
		case Action::Append: s += "a"; break;
		}

		if (start2 == end2)
			s += to_string(start2);
		else
			s += to_string(start2) + "," + to_string(end2);

		return s;
	}
};


class TLCS_Finder {
public:
	TLCS_Finder(const TFile & file1, const TFile & file2, bool ignoreNL=false);
	void PrintEditScript();
	void PrintLine(int fileNumber, TLineNumber i);
	~TLCS_Finder();
	bool ignoreNewLines;

private:

	TLineNumber m, n;
	const TFile & f1;
	const TFile & f2;

	Step **b;
	long long **c;

	TLineNumber gaps1 = 0, gaps2 = 0;
	int start1 = 0, start2 = 0, end1 = 0, end2 = 0;

	vector <TDiffAction> actions;

	void FindAlignment();
	bool CompareLines(TLineNumber i, TLineNumber j);
	void FindEditScript();
	void AddActionToEditScript(TLineNumber i, TLineNumber j);

	//for debug
	void Print_LCS();
	void Print_LCS_Rec(TLineNumber i, TLineNumber j);
	void PrintDynProgTable();
};

#endif
