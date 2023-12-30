#include <iostream>
#include <stack>
#include <vector>  

size_t CalculateCurrentArea(const std::stack <size_t> &stack, size_t pos) {
	if (!stack.empty()) {
		return pos - stack.top() - 1; 
	}
	return pos;                     
}                          

size_t MaxAreaHistogram(const std::vector<size_t> &arr) {
	std::stack <size_t> stack;
	size_t area_max = 0;
	size_t area_current = 0;
	
	size_t i = 0;

	while (i < arr.size() || !stack.empty()) {
		if (i < arr.size() && (stack.empty() || arr[stack.top()] <= arr[i])) {
			stack.push(i);
			i++;
		} else {
			size_t stack_top = stack.top();
			stack.pop();
			area_current = arr[stack_top] * CalculateCurrentArea(stack, i);
			std::cout << arr[stack_top] << ' ' << CalculateCurrentArea(stack, i) << ' ' << area_current << std::endl;
			if (area_current > area_max) {
				area_max = area_current;
			}
		}
	}

	return area_max;
}                                                                                    

int main(void) {         
	size_t height;
	size_t width;
	std::cin >> height >> width;
	char symbol;

	std::vector<size_t> tmp(width, 0);

	size_t res = 0;
	for (size_t i = 0; i < height; i++) {       
		for (size_t j = 0; j < width; j++) {   
		    std::cin >> symbol;
			if (symbol == '0') {                
				tmp[j]++;
			} else {                            				 
				tmp[j] = 0;
			}
		}
		
		size_t res_tmp = MaxAreaHistogram(tmp);
		if (res_tmp > res) {                    
			res = res_tmp;              
		}
	}              

	std::cout << res << std::endl;

	return 0;
}
