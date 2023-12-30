#include <iostream>
#include <vector>
#include <ctime>
#include <stack>

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
			//std::cout << arr[stack_top] << ' ' << CalculateCurrentArea(stack, i) << ' ' << area_current << std::endl;
			if (area_current > area_max) {
				area_max = area_current;
			}
		}
	}

	return area_max;
}


size_t NaiveMethod(const std::vector<std::vector<char> > &data) {
    size_t height = data.size();
    size_t width  = data[0].size(); 
    size_t max_area = 0;
    for(size_t x0 = 0; x0 < height; x0++) {
        for(size_t y0 = 0; y0 < width; y0++) {
            for(size_t x1 = x0 + 1; x1 < height; x1++) {
                for(size_t y1 = y0 + 1; y1 < width; y1++) {
                    bool allzeroes = true;
                    for(size_t i = x0; i <= x1; i++) {
                        for(size_t j = y0; j <= y1; j++) {
                            if(data[i][j] == '1') {
                                allzeroes = false;
                                break;
                            }
                        }
                    }
                    if (allzeroes) {
                        if ((x1 - x0 + 1) * (y1 - y0 + 1) > max_area)
                            max_area = (x1 - x0 + 1) * (y1 - y0 + 1);
                    }
                }
            }
        }
    }
    
    return max_area;
}

int main(void) {
  	size_t height;
    size_t width;
	std::cin >> height >> width;
    std::vector<std::vector<char> > data(height);

	for (size_t i = 0; i < height; i++) {
	    data[i].resize(width);      
		for (size_t j = 0; j < width; j++) {   
		    std::cin >> data[i][j];
		}
	}  



	
		
    clock_t t0 = clock(), t;
    size_t res = 0;
    std::vector<size_t> tmp(width, 0);
	for (size_t i = 0; i < height; i++) {       
		for (size_t j = 0; j < width; j++) {   
			if (data[i][j] == '0') {                
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

	std::cout << "Max. area    (DP) = " << res << std::endl;
	
    t = clock();
    std::cout << "Time         (DP) = " << std::fixed << (double)(t - t0) / (double)CLOCKS_PER_SEC << std::endl;
    
    t0 = clock();
    std::cout << "Max. area (naive) = " << std::fixed << NaiveMethod(data) << std::endl;
    t = clock();
    std::cout << "Time      (naive) = " << (double)(t - t0) / (double)CLOCKS_PER_SEC << std::endl;
      
    return 0;
}
