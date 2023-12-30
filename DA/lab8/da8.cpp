#include <iostream>
#include <math.h>

int * CountSort(int max_segment, int * cur_segment, size_t n) {
    int * counts = new int[max_segment + 1];
    int * arr    = new int[n];
    
    for (int i = 0; i <= max_segment; i++) {
        counts[i] = 0;
    }
    
    for (int i = 0; i < n; i++) {
        counts[cur_segment[i]]++; 
    }
    
    for (int i = 1; i <= max_segment; i++) {
	    counts[i] += counts[i - 1];
    }

    for (int i = n - 1; i >= 0; i--) {
    	arr[counts[cur_segment[i]] - 1] = cur_segment[i];
    	counts[cur_segment[i]]--;
    }
    
    delete [] counts;
    
    return arr;
}


bool SimpleCheck(size_t a, size_t b, size_t c) {
    return a + b > c;
}

double Area(size_t a, size_t b, size_t c) {
    double p = (a + b + c) / 2.0;
    return sqrt(p * (p - a) * (p - b) * (p - c));
}

bool FindMaxSquare(int * arr, size_t n) {
	double area = 0, area2 = 0;
	size_t ii = 0;
	for (int i = n - 3; i >= 1; i--) {
	    if(SimpleCheck(arr[i], arr[i + 1], arr[i + 2])) {
			area = Area(arr[i], arr[i + 1], arr[i + 2]);
			ii = i;
		}
		
		if (SimpleCheck(arr[i - 1], arr[i], arr[i + 1])) {
			area2 = Area(arr[i - 1], arr[i], arr[i + 1]);
			if (area2 > area) {
			    area = area2;
			    ii = i - 1;
			}
		}
		
		if(area != 0) {
		    std::cout.precision(3);
		    std::cout << std::fixed << area << std::endl;
		    std::cout << arr[ii] << " " << arr[ii + 1] << " " << arr[ii + 2]; 
		    return true;
	    }	
	}
	return false;
}

int main() {
	
	size_t n;
	std::cin >> n;
	
	if (n < 3) {
		std::cout << "0" << std::endl;
		return 0;
	}
	
	int max_segment = 0;
	int * cur_segment = new int[n];
	for (int i = 0; i < n; i++) {
		std::cin >> cur_segment[i];
		if (cur_segment[i] > max_segment)
			max_segment = cur_segment[i]; 
	}
	
	int * res = CountSort(max_segment, cur_segment, n); 	

	bool found_triangle = FindMaxSquare(res, n);
	if (!found_triangle)
	    std::cout << "0" << std::endl;
	
	delete [] cur_segment;
	delete [] res;
	return 0;
}
