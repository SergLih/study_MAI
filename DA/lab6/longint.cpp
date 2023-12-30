#include "longint.h"

TLongInt::TLongInt() {
	digits.push_back(0);
}

TLongInt::TLongInt(const std::string &str) {
	int s_pos = str.size() - 1;
	while (s_pos >= 0) {
		int longDigit = 0, p = 1;
		while (s_pos >= 0 && p != BASE) {
			longDigit += (str[s_pos] - '0') * p;
			p *= 10;
			s_pos--;
		}
		digits.push_back(longDigit);
	}
	DeleteZeros();
}

TLongInt::TLongInt(const TDigit & digit) {
	digits.push_back(digit);
}

TLongInt::TLongInt(const size_t &size) {
	digits.resize(size, 0);
}

TLongInt TLongInt::Power(const TLongInt &pow) const {
	TLongInt zero, one("1"), two("2");
	if (pow == zero) 
		return one;
    else if (pow == one)
	    return *this;
	    
	if (*this == one) 
	    return one;
	
	if (pow.digits[0] % 2 == 0) {
		TLongInt res = this->Power(pow / two);
		return res * res;
	} else {
		TLongInt res = pow - one;
		res = this->Power(res);
		return (*this) * res;
	}
}

void TLongInt::DeleteZeros() {
	while (digits.size() > 1 && digits.back() == 0) {
		digits.pop_back();
	}
}

TLongInt TLongInt::FromDigits(const size_t i, const size_t j) {
	TLongInt res(j - i + 1);
	for (size_t k = 0; k < res.size(); k++) {
		res.digits[k] = digits[i + k];
	}
	return res;
}

void TLongInt::ReplaceDigits(TLongInt & other, size_t i, size_t j) {
	for (int k = 0; k < other.size(); k++) {
		digits[i + k] = other.digits[k];
	}
	j = std::min(j, this->size()-1);	//защита
	for (int k = other.size() + i; k <= j; k++) {
		digits[k] = 0;
	}
}

void TLongInt::PrepareBeforeSafeSubstraction() {
	this->digits.push_back(1);
}

void TLongInt::PrepareAfterSafeSubstraction() {
	this->digits.pop_back();
}

TLongInt TLongInt::operator+(const TLongInt &other) const {
	TDigit newDigit = 0;
	size_t maxLen = std::max(digits.size(), other.size()) + 1;
	TLongInt result(maxLen);
	for (size_t i = 0; i < maxLen; ++i) {
		if (i < digits.size())
			newDigit += digits[i];
		if (i < other.digits.size())
			newDigit += other.digits[i];

		result.digits[i] = newDigit % BASE;
		newDigit /= BASE;
	}
	result.DeleteZeros();
	return result;
}

TLongInt TLongInt::operator-(const TLongInt &other) const {
	TDigit newDigit = 0;
	TDigit borrowed = 0;
	TLongInt result(digits.size());
	for (int i = 0; i < digits.size(); ++i) {
		newDigit = digits[i] - (i < other.size() ? other.digits[i] : 0) + borrowed;
		if (newDigit < 0) {
			borrowed = -1;
			newDigit += BASE;
		} else {
			borrowed = 0;
		}
		result.digits[i] = newDigit;
	}
	result.DeleteZeros();
	return result;
}

TLongInt TLongInt::operator*(const TLongInt &other) const {
	size_t size1 = size();
	size_t size2 = other.size();
	TLongInt result(size1 + size2);
	for (int j = 0; j < size2; ++j) {
		if (other.digits[j] == 0) 
			continue;
		
		for (int i = 0; i < size1; ++i) {
			result.digits[i + j] += digits[i] * other.digits[j];
			result.digits[i + j + 1] += result.digits[i + j] / BASE;
			result.digits[i + j] %= BASE;
		}
	}
	result.DeleteZeros();
	return result;
}

TLongInt TLongInt::operator/(TLongInt &other) const {
    
    if (*this < other) {
        return TLongInt();
    }

	TDigit d = BASE / (other.digits.back() + 1); //D1
	TLongInt u = *this * d;
	u.digits.resize(this->size()+1, 0);

	TLongInt v = other * d;

	size_t uSize = u.size();
	size_t vSize = v.size();

	TLongInt q(uSize - vSize + 1);
	TLongInt r;
	for (int j = this->size() - vSize; j >= 0; j--) { //D2-D7
		//D3
		TDigit q_hat = (u.digits[j + vSize] * BASE 
			+ u.digits[j + vSize - 1]) / v.digits.back();
		if (q_hat == BASE) {
			q_hat--;
		}
		//D4
				
		TLongInt minuend = u.FromDigits(j, j + vSize);
		minuend.PrepareBeforeSafeSubstraction();
		TLongInt res = minuend - (TLongInt(q_hat) * v);
		//проверяем на "отрицательный" результат с помощью кол-ва цифр:
		while (res.size() != minuend.size()) {
			q_hat--;
			res = res + v;
		}

		res.PrepareAfterSafeSubstraction();
		u.ReplaceDigits(res, j, j + vSize);

		q.digits[j] = q_hat; //D5
		
	}
	q.DeleteZeros();
	return q;
}

bool TLongInt::operator<(const TLongInt &other) const {
	if (size() != other.size()) {
		return size() < other.size();
	}
	for (int i = size() - 1; i >= 0; --i) {
		if (digits[i] != other.digits[i]) {
			return digits[i] < other.digits[i];
		}
	}
	return false;
}

bool TLongInt::operator>(const TLongInt &other) const {
	if (size() != other.size()) {
		return size() > other.size();
	}
	for (int i = size() - 1; i >= 0; --i) {
		if (digits[i] != other.digits[i]) {
			return digits[i] > other.digits[i];
		}
	}
	return false;
}

bool TLongInt::operator==(const TLongInt &other) const {
	if (size() != other.size()) 
		return false;
	
	for (size_t i = 0; i < size(); i++)
		if (digits[i] != other.digits[i]) 
			return false;
		
	return true;
}

std::ostream & operator<< (std::ostream& stream, const TLongInt & number) {
	if (number.size() == 0) {
		return stream;
	}
	stream << number.digits[number.size() - 1];
	for (int i = number.size() - 2; i >= 0; --i) {
		stream << std::setfill('0') << std::setw(R) << number.digits[i];
	}
	return stream;
}
