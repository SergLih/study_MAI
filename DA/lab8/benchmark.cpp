#include <iostream>
#include <ctime>
#include <math.h>

size_t * CountSort(size_t * arr_sides, size_t n, size_t max_segment) {

    size_t * counts = new size_t[max_segment + 1];
    size_t * arr    = new size_t[n];
    
    for (size_t i = 0; i <= max_segment; i++) {
        counts[i] = 0;
    }
    
    for (size_t i = 0; i < n; i++) {
        counts[arr_sides[i]]++; 
    }
    
    for (size_t i = 1; i <= max_segment; i++) {
	    counts[i] += counts[i - 1];
    }

    for (int i = n - 1; i >= 0; i--) {
    	arr[counts[arr_sides[i]] - 1] = arr_sides[i];
    	counts[arr_sides[i]]--;
    }
    
    delete [] counts;
    
    return arr;
}


bool SimpleCheck(size_t a, size_t b, size_t c) {
    return a + b > c;
}

bool FullCheck(size_t a, size_t b, size_t c) {
    return a + b > c && a + c > b && b + c > a;
}

double Area(size_t a, size_t b, size_t c) {
    double p = (a + b + c) / 2.0;
    return sqrt(p * (p - a) * (p - b) * (p - c));
}

bool FindMaxSquare_Greedy(size_t * arr_sides, size_t n, size_t max_segment, size_t *a, size_t *b, size_t * c, double * area) {
    size_t * arr = CountSort(arr_sides, n, max_segment); 	
	*area = 0;
	double area2 = 0;
	size_t ii = 0;
	for (int i = n - 3; i >= 1; i--) {
	    if(SimpleCheck(arr[i], arr[i + 1], arr[i + 2])) {
			*area = Area(arr[i], arr[i + 1], arr[i + 2]);
			ii = i;
		}
		
		if (SimpleCheck(arr[i - 1], arr[i], arr[i + 1])) {
			area2 = Area(arr[i - 1], arr[i], arr[i + 1]);
			if (area2 > *area) {
			    *area = area2;
			    ii = i - 1;
			}
		}
		
		if(*area != 0) {
		    *a = arr[ii];
		    *b = arr[ii + 1];
		    *c = arr[ii + 2];
		    delete[] arr;
		    return true;
	    }	
	}
	delete[] arr;
	return false;
}

bool FindMaxSquare_Naive(size_t * arr, size_t n, size_t *a, size_t *b, size_t * c, double * area) { 
    double cur_area = 0.0;
    *area = 0.0;
    for (size_t i = 0; i < n - 2; i++) {
        for (size_t j = i + 1; j < n - 1; j++) {
            for (size_t k = j + 1; k < n; k++) {
                if (FullCheck(arr[i], arr[j], arr[k])) {
			        cur_area = Area(arr[i], arr[j], arr[k]);
			        if (cur_area > *area) {
			            *area = cur_area;
			            *a = arr[i];
			            *b = arr[j];
			            *c = arr[k];
			        }
	            }
            }
        }
    }
    
    return (*area != 0);
}

int main(void) {
    size_t n;
	size_t a, b, c;
	double area;
	std::cin >> n;
	
	if (n < 3) {
		std::cout << "0" << std::endl;
		return 0;
	}
	std::cout.precision(6);
	
	size_t max_segment = 0;
	size_t * cur_segment = new size_t[n];
	for (size_t i = 0; i < n; i++) {
		std::cin >> cur_segment[i];
		if (cur_segment[i] > max_segment)
			max_segment = cur_segment[i]; 
	}
		
    clock_t t0 = clock(), t;

	bool found_triangle = FindMaxSquare_Greedy(cur_segment, n, max_segment, &a, &b, &c, &area);
	if (!found_triangle)
	    std::cout << "0" << std::endl;
	else
		std::cout << std::fixed << area << std::endl << a << " " << b << " " << c << std::endl;    

    t = clock();
    std::cout << "Time         (GA) = " << std::fixed << (double)(t - t0) / (double)CLOCKS_PER_SEC << std::endl;
    
    t0 = clock();
    
    found_triangle = FindMaxSquare_Naive(cur_segment, n, &a, &b, &c, &area);
	if (!found_triangle)
	    std::cout << "0" << std::endl;
	else
		std::cout << std::fixed << area << std::endl << a << " " << b << " " << c << std::endl;   
    
    t = clock();
    std::cout << "Time      (naive) = " << (double)(t - t0) / (double)CLOCKS_PER_SEC << std::endl;
      

   
    delete [] cur_segment;
    return 0;
}
