#include "mystring.h"

TString::TString() {
	this->length = 0;
	this->code = nullptr;
}

TString::TString(char * _cstring) {
	Encode(_cstring);
}

TString::~TString() {
	if (code != nullptr)
		delete[] code;
	code = nullptr;
}

TString & TString::operator=(const TString & other) {
	
	int blocks = BlocksCnt();
	if (length != other.length) {
	    length = other.length;
	    if (code != nullptr)
		    delete[] code;
		blocks = BlocksCnt();
	    code = new long long[blocks];
	}
	for (size_t i = 0; i < blocks; i++) {
		code[i] = other.code[i];
	}
	return *this;
}

std::istream & operator>>(std::istream & is, TString & s) {
	char buf[STR_MAX_SIZE];
	is >> buf;
	s.Encode(buf);
	return is;
}

std::ostream & operator<<(std::ostream & os, const TString & s) {
	os << "Not implemented\n";
	return os;
}

void Serialize(const TString &obj, std::ofstream & file) {
	unsigned char ser_length = (unsigned char)(obj.length - 1);  //0-255 -> 1-256
	file.write((char *)&(ser_length), sizeof(ser_length));

	int blocks = obj.BlocksCnt();
	for (size_t i = 0; i < blocks; i++) {
		file.write((char*) &obj.code[i], sizeof(long long));
	}
}

bool Deserialize(TString &obj, std::ifstream & file) {
	if (file) {
		unsigned char tmp_length2;
		file.read((char *)&tmp_length2, sizeof(tmp_length2));
		size_t tmp_length = tmp_length2;  //0-255 -> 1-256
		int blocks = obj.BlocksCnt();
		if(obj.length != tmp_length + 1) {
		    obj.length = tmp_length + 1;
		    if (obj.code != nullptr)
			    delete[] obj.code;
			blocks = obj.BlocksCnt();
		    obj.code = new long long[blocks];
		}
		for (size_t i = 0; i < blocks; i++) {
			file.read((char*)&obj.code[i], sizeof(long long));
		}
		return true;
	}
	else {
		std::cerr << "error: only " << file.gcount() << " bytes could be read";
		return false;
	}
}

void TString::Encode(const char * s) {
	size_t len = strlen(s);
	int blocks = BlocksCnt();
	
	if(length != len) {
	    length = len;
	    if (code != nullptr)
		    delete[] code;
		blocks = BlocksCnt();
	    code = new long long[blocks];
	}

	int j = len - 1;
	for (size_t i = 0; i < blocks; i++)	{
		unsigned long long t = 0;
		unsigned long long pow = 1;
		for (size_t k = 0; k < 13 && j >= 0; k++) {
			t += (tolower(s[j--]) - 'a') * pow;
			pow *= 26;
		}
		code[i] = t;
	}
}

int TString::BlocksCnt() const {
	return (length + 12) / 13;     //+12 -- ceil()
}

int CompareKey(const TString &k1, const TString &k2) {
	int cmp = k1.length - k2.length;
	if (cmp != 0)
		return cmp;
	else {
		int blocks = k1.BlocksCnt();
		for (size_t i = 0; i < blocks; i++) {
			long long cmp2 = k1.code[i] - k2.code[i];
			cmp = (cmp2 > 0) - (cmp2 < 0);
			if (cmp != 0)
				return cmp;
		}
		return 0;
	}
}
