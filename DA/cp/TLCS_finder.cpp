#include "TLCS_finder.h"

TLCS_Finder::TLCS_Finder(const TFile & file1, const TFile & file2, bool ignoreNL) : f1(file1), f2(file2) {
	m = file1.lines.size();
	n = file2.lines.size();

	b = new Step*[m + 1];
	c = new long long*[m + 1];
	for (TLineNumber i = 0; i <= m; i++) {
		b[i] = new Step[n + 1];
		c[i] = new long long[n + 1];
	}
	ignoreNewLines = ignoreNL;
	FindAlignment();
	FindEditScript();
}

void TLCS_Finder::PrintEditScript() {
	for (TDiffAction &act : actions) {
		cout << act.ToString() << endl;
		switch (act.action) {
		case Action::Delete:
			for (TLineNumber j = act.start1 - 1; j < act.end1; j++)
				PrintLine(1, j);
			break;
		case Action::Append:
			for (TLineNumber j = act.start2 - 1; j < act.end2; j++)
				PrintLine(2, j);
			break;
		case Action::Change:
			for (TLineNumber j = act.start1 - 1; j < act.end1; j++)
				PrintLine(1, j);
			cout << "---\n";
			for (TLineNumber j = act.start2 - 1; j < act.end2; j++)
				PrintLine(2, j);
			break;
		default:
			break;
		}
	}
}

void TLCS_Finder::PrintLine(int fileNumber, TLineNumber i) {
	if (fileNumber == 1) {
		cout << "< " << f1.lines[i] << "\n";
		if (i == f1.lines.size() - 1 && f1.noNewLineAtEOF)
			cout << "\\ No newline at end of file\n";
	}
	else {
		cout << "> " << f2.lines[i] << "\n";
		if (i == f2.lines.size() - 1 && f2.noNewLineAtEOF)
			cout << "\\ No newline at end of file\n";
	}
}

TLCS_Finder::~TLCS_Finder() {
	for (TLineNumber i = 0; i <= m; i++) {
		delete[] b[i];
		delete[] c[i];
	}
	delete[] b;
	delete[] c;
}

void TLCS_Finder::FindAlignment() {
	for (TLineNumber i = 1; i <= m; i++) {
		b[i][0] = Step::Up;
		c[i][0] = 0;
	}
	for (TLineNumber j = 0; j <= n; j++) {
		b[0][j] = Step::Left;
		c[0][j] = 0;
	}
	for (TLineNumber i = 1; i <= m; i++) {
		for (TLineNumber j = 1; j <= n; j++) {
			if (CompareLines(i, j)/*f1.lines[i - 1] == f2.lines[j - 1]*/) {
				c[i][j] = c[i - 1][j - 1] + 1;
				b[i][j] = Step::UpLeft;
			}
			else if (c[i - 1][j] >= c[i][j - 1]) {
				c[i][j] = c[i - 1][j];
				b[i][j] = Step::Up;
			}
			else {
				c[i][j] = c[i][j - 1];
				b[i][j] = Step::Left;
			}
		}
	}
}

bool TLCS_Finder::CompareLines(TLineNumber i, TLineNumber j) {
	bool strings_equal = (f1.lines[i-1] == f2.lines[j-1]);

	if(!strings_equal)
		return false;

	if (ignoreNewLines)
		return true;

	if (i == m && j == n)
		return f1.noNewLineAtEOF == f2.noNewLineAtEOF;
	else if (i == m)
		return !f1.noNewLineAtEOF;
	else if (j == n)
		return !f2.noNewLineAtEOF;
	else
		return true;
}

void TLCS_Finder::AddActionToEditScript(TLineNumber i, TLineNumber j) {
	if (gaps1 == 0 && gaps2 > 0) {
		actions.insert(actions.begin(),
			TDiffAction(Action::Delete, start1, end1, j, j));
	}
	else if (gaps1 > 0 && gaps2 == 0) {
		actions.insert(actions.begin(),
			TDiffAction(Action::Append, i, i, start2, end2));
	}
	else if (gaps1 > 0 && gaps2 > 0) {
		actions.insert(actions.begin(),
			TDiffAction(Action::Change, start1, end1, start2, end2));
	}

	start1 = start2 = end1 = end2 = gaps1 = gaps2 = 0;
}

void TLCS_Finder::FindEditScript() {

#ifdef DEBUG
	string aln1 = "";
	string aln2 = "";
#endif

	TLineNumber i = m, j = n;
	while (i != 0 || j != 0) {
		if (b[i][j] == Step::UpLeft) {
			AddActionToEditScript(i, j);
#ifdef DEBUG
			aln1 += to_string(i) + "\t";
			aln2 += to_string(j) + "\t";
#endif
			i--;
			j--;
		}
		else if (b[i][j] == Step::Up) {
			gaps2++;
			if (end1 == 0)
				start1 = end1 = i;
			else
				start1 = i;
#ifdef DEBUG
			aln1 += to_string(i) + "\t";
			aln2 += "-\t";
#endif
			i--;
		}
		else {
			gaps1++;
			if (end2 == 0)
				start2 = end2 = j;
			else
				start2 = j;
#ifdef DEBUG
			aln1 += "-\t";
			aln2 += to_string(j) + "\t";
#endif
			j--;
		}
	}
	AddActionToEditScript(0, 0);
#ifdef DEBUG
	std::reverse(aln1.begin(), aln1.end());
	std::reverse(aln2.begin(), aln2.end());
	cout << aln1 << "\n" << aln2 << "\n";
#endif
}

void TLCS_Finder::PrintDynProgTable() {
	for (TLineNumber i = 0; i <= m; i++) {
		for (TLineNumber j = 0; j <= n; j++) {
			if (b[i][j] == Step::UpLeft)
				cout << " \\ ";
			else if (b[i][j] == Step::Left)
				cout << " - ";
			else if (b[i][j] == Step::Up)
				cout << " | ";
			else
				cout << " * ";
		}
		cout << "\n";
	}
}

void TLCS_Finder::Print_LCS() {
	Print_LCS_Rec(m, n);
	cout << endl;
}

void TLCS_Finder::Print_LCS_Rec(TLineNumber i, TLineNumber j) {
	if (i == 0 || j == 0) {
		return;
	}
	if (b[i][j] == Step::UpLeft) {
		Print_LCS_Rec(i - 1, j - 1);
		cout << f1.lines[i - 1];
	}
	else if (b[i][j] == Step::Up) {
		Print_LCS_Rec(i - 1, j);
	}
	else {
		Print_LCS_Rec(i, j - 1);
	}
}
