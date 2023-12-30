#ifndef LONGINT_H
#define LONGINT_H

#include <vector>
#include <iostream>
#include <cmath>
#include <iomanip>

typedef long long TDigit;

const int R = 8;
const int BASE = pow(10, R);

class TLongInt {
public:
	TLongInt();
    TLongInt(const std::string &str);
	TLongInt(const TDigit &digit);
    TLongInt(const size_t &size);
    ~TLongInt() {};
    inline size_t size() const { return digits.size(); }

    TLongInt operator+(const TLongInt &other) const;
    TLongInt operator-(const TLongInt &other) const;
    TLongInt operator*(const TLongInt &other) const;
    TLongInt operator/(TLongInt &other) const;
    TLongInt Power(const TLongInt &pow) const;
    bool operator<(const TLongInt &other) const;
    bool operator>(const TLongInt &other) const;
    bool operator==(const TLongInt &other) const;
    friend std::ostream& operator<< (std::ostream& stream, const TLongInt & number);

private:
	std::vector<TDigit> digits;
	void DeleteZeros();
	TLongInt FromDigits(const size_t i, const size_t j);
	void ReplaceDigits(TLongInt &other, size_t i, size_t j);
	void PrepareBeforeSafeSubstraction();
	void PrepareAfterSafeSubstraction();
};

#endif
