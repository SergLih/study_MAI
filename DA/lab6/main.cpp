#include "longint.h"

int main() {
    std::string s_num1, s_num2;
    std::string action;
    while(std::cin >> s_num1 >> s_num2 >> action) {
        bool cmp = false, err = false;
        TLongInt num1(s_num1), num2(s_num2), res;
        if(action == "+") {
            res = num1 + num2;
        } else if(action == "-") {
             if(num1 < num2) {
                err = true;
             } else {
                res = num1 - num2;
             }
        } else if(action == "*") {
            res = num1 * num2;
        } else if(action == "/") {
            if(num2 == TLongInt("0")) {
                err = true;
            } else {
                res = num1 / num2;
            }
        } else if(action == "^") {
            if(num1 == TLongInt("0") && num2 == TLongInt("0")) {
                err = true;
            } else {
                res = num1.Power(num2);
            }
        } else {
            cmp = true;
            if(action == ">") {
                num1 > num2 ? std::cout << "true" << std::endl : std::cout << "false" << std::endl;
            } else if(action == "<") {
                num1 < num2 ? std::cout << "true" << std::endl : std::cout << "false" << std::endl;
            } else if(action == "=") {
                num1 == num2 ? std::cout << "true" << std::endl : std::cout << "false" << std::endl;
            }
        }
        if(!cmp) {
            if(err) {
                std::cout << "Error" << std::endl;
            } else {
                std::cout << res << std::endl;
            }
        }
    }
    return 0;
}
