#ifndef __MYSTRING_H__
#define __MYSTRING_H__

#include "main.h"

const int STR_MAX_SIZE = 512;

class TString {
private:
	size_t length;
	long long *code;

	void Encode(const char * src);
	int BlocksCnt() const;
public:
	TString();
	TString(char * cstring);
	~TString();

	TString & operator=(const TString & other);

	friend std::istream & operator>>(std::istream & is, TString &s);
	friend std::ostream & operator<<(std::ostream & os, const TString &s);

	friend int CompareKey(const TString &k1, const TString &k2);
	friend void Serialize(const TString &obj, std::ofstream &file);
	friend bool Deserialize(TString &obj, std::ifstream &file);
};

int CompareKey(const TString &k1, const TString &k2);
void Serialize(const TString &obj, std::ofstream &file);
bool Deserialize(TString &obj, std::ifstream &file);

#endif // __MYSTRING_H__
