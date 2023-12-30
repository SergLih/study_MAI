#include <stdlib.h>
#include <iostream>
#include <string>
#include <cstring>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;


int ind(int i, int j, int k, int wx, int wy) {
	return i + wx * ( j + wy * k );
}


double* CreateMatrix3D(int n_x, int n_y, int n_z, double u_down, double u_up, double u_left, double u_right, double u_front, double u_back, double u_0) {
	double *data = new double[(n_x+2)*(n_y+2)*(n_z+2)];
	for (int i = 0; i <= n_x+1; ++i){
		for (int j = 0; j <= n_y+1; ++j){
			for (int k = 0; k <= n_z+1; ++k){
				int local1d = ind(i, j, k, n_x+2, n_y+2);

				if(i == 0)
					data[local1d] = u_left;
				else if(i == n_x+1)
					data[local1d] = u_right;
				if (j == 0)
					data[local1d] = u_front;
				else if(j == n_y+1)
					data[local1d] = u_back;
				if (k == 0)
					data[local1d] = u_down;
				else if(k == n_z+1)
					data[local1d] = u_up;
				
				if(i*j*k > 0 && (n_x+1-i)*(n_y+1-j)*(n_z+1-k)>0)
					data[local1d] = u_0;

			}
		}
	}	
	
	return data;
}



int main (int argc, char *argv[]) {

// На первой строке заданы три числа: размер сетки
// процессов. Гарантируется, что при запуске программы количество процессов будет
// равно произведению этих трех чисел. На второй строке задается размер блока,
// который будет обрабатываться одним процессом: три числа. Далее задается путь к
// выходному файлу, в который необходимо записать конечный результат работы
// программы и точность ε . На последующих строках описывается задача: задаются
// размеры области lx , ly и lz , граничные условия: udown
// , uup , uleft , uright , ufront и uback , и начальное значение u .
 
 	int n_proc_x, n_proc_y, n_proc_z;
 	int bsz_x, bsz_y, bsz_z;
 	string filename;
 	double precision, max_error_iter;
 	double lx, ly, lz;
 	double u_down, u_up, u_left, u_right, u_front, u_back;
 	double u_0;

		cin >> n_proc_x >> n_proc_y >> n_proc_z;
 	cin >> bsz_x >> bsz_y >> bsz_z;
 	cin >> filename;
 	cin >> precision >> lx >> ly >> lz;
 	cin >> u_down >> u_up >> u_left >> u_right >> u_front >> u_back >> u_0;

 	cerr << n_proc_x << " " << n_proc_y << " " << n_proc_z << " " << endl;
 	cerr << bsz_x << " " << bsz_y << " " << bsz_z << endl;
 	cerr << precision << " " << lx << " " << ly << " " << lz << endl;
 	cerr << u_down << " " << u_up << " " << u_left << " " << u_right << " " << u_front << " " << u_back << " " << u_0 << endl;


	int n_x, n_y, n_z, n_total;
	n_x = n_proc_x * bsz_x;
	n_y = n_proc_y * bsz_y;
	n_z = n_proc_z * bsz_z;
	n_total = (n_x+2)*(n_y+2)*(n_z+2);

	double h_x, h_y, h_z;
	h_x = pow(lx / n_x, -2);
	h_y = pow(ly / n_y, -2);
	h_z = pow(lz / n_z, -2);

	double *data = CreateMatrix3D(n_x, n_y, n_z, u_down, u_up, u_left, u_right, u_front, u_back, u_0);

	time_t start0 = clock();
	int iter = 0;
	do {
		iter++;

		double * new_data = new double[n_total];
		memcpy(new_data, data, sizeof(double)*n_total);

		max_error_iter = 0;
		for (int i = 1; i <= n_x; ++i) {
			for (int j = 1; j <= n_y; ++j) {
				for(int k = 1; k <= n_z; ++k) {
					new_data[ind(i, j, k, n_x+2, n_y+2)] =
								 ((data[ind(i+1, j,   k,   n_x+2, n_y+2)] 
								+  data[ind(i-1, j,   k,   n_x+2, n_y+2)]) *h_x 
								+ (data[ind(i,   j+1, k,   n_x+2, n_y+2)] 
								+  data[ind(i,   j-1, k,   n_x+2, n_y+2)]) *h_y
								+ (data[ind(i,   j,   k+1, n_x+2, n_y+2)] 
								+  data[ind(i,   j,   k-1, n_x+2, n_y+2)]) *h_z)
								/ (2*(h_x + h_y + h_z)); 
					double error = abs(new_data[ind(i, j, k, n_x+2, n_y+2)] -
						            data[ind(i, j, k, n_x+2, n_y+2)] );
					if(error > max_error_iter)
						max_error_iter = error;
				}
			}
		}

		memcpy(data, new_data, sizeof(double)*n_total);
		delete[] new_data;
		
		//cerr << "i " + to_string(iter) + ": e=" + to_string(max_error_iter) + "\t";
	
	} while(max_error_iter > precision);
    time_t end0 = clock();
    
    fprintf(stderr, "CPU: ready\n");
    fprintf(stderr, "Working time:     %f sec.", 
        (double)(end0 - start0) / (double)CLOCKS_PER_SEC);



	ofstream resout(filename, std::ofstream::out);
		for (int k = 1; k <= n_z; ++k) {
			for (int j = 1; j <= n_y; ++j) {
				for (int i = 1; i <= n_x; ++i) {
					resout << scientific << setprecision(6) 
					       << data[ind(i, j, k, n_x+2, n_y+2)] << " ";
				}
			resout << "\n";
		}	
		resout << "\n";
	}
	resout.close();
	

}
