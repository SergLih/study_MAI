#include <stdlib.h>
#include <iostream>
#include <string>
#include <cstring>
#include "mpi.h"
#include <iomanip>
#include <cmath>
#include <fstream>
#include <sstream>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

using namespace std;

#define CSC(call) do { \
	cudaError_t pixels = call;	\
	if (pixels != cudaSuccess) { \
		fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(pixels)); \
		exit(0); \
	} \
} while (0)


enum BorderDir {
	LeftToRight, RightToLeft,
	UpToDown, DownToUp,
	FrontToBack, BackToFront,
	Out,
};

__host__ __device__ int ind(int i, int j, int k, int wx, int wy) {
	return i + wx * ( j + wy * k );
}


// string get_pid(int pid, int n_proc_x, int n_proc_y)
// {
// 	int idx = pid;
// 	int pz = idx / (n_proc_x * n_proc_y);
//     idx -= (pz * n_proc_x * n_proc_y);
//     int py = idx / n_proc_x;
//     int px = idx % n_proc_x;
//     return "P_" + to_string(pid) + "[" + to_string(px) + " " + to_string(py) + " " + to_string(pz) + "] ";
// }

// string printOutput(int pid, int n_proc_x, int n_proc_y, double *data_h, int bsz_x, int bsz_y, int bsz_z, int iter) {
// 	//ofstream fout(fn, std::ofstream::out);
// 	ostringstream oss;
// 	oss << get_pid(pid, n_proc_y, n_proc_y) + ": " + to_string(iter) << endl;
// 	//fout << fn << endl;
// 	for (int k = 0; k < bsz_z+2; ++k){
// 		for (int j = 0; j < bsz_y+2; ++j) {
// 			for (int i = 0; i < bsz_x+2; ++i) {
// 				oss << scientific << setprecision(6) << data_h[ind(i, j, k, bsz_x+2, bsz_y+2)] << "\t";
// 			}
// 			oss << "\n";
// 		}
// 		oss << "\n==========================\n";
// 	}
// 	return oss.str();
// }

// void print(int bsz_x, int bsz_y, int bsz_z, double *data_h, ofstream &resout) {
// 	for (int k = 1; k <= bsz_z; ++k) {
// 			for(int j = 1; j <= bsz_y; ++j) {
// 				for (int i = 1; i <= bsz_x; ++i) {
// 					resout << scientific << setprecision(6) << data_h[ind(i, j, k, bsz_x+2, bsz_y+2)] << " "; 
// 				}
// 				resout << "\n";
// 			}
// 			resout << "\n";
// 		}
// }

struct cmpAbsDoubles {
	__host__ __device__ bool operator()(double a, double b) {
		return fabs(a) < fabs(b);
	}
};

__global__ void jacobi_kernel(double *data_d, double * new_data_d, 
							  int bsz_x, int bsz_y, int bsz_z, double h_x, double h_y, double h_z) {
	int tidx = threadIdx.x;;
	int tidy = threadIdx.y;
	int tidz = threadIdx.z;
	int ofsx = blockDim.x;
	int ofsy = blockDim.y;
	int ofsz = blockDim.z;
	//for(int k = 0; k <= bsz_z+1; ++k)
	for(int k = tidz; k <= bsz_z + 1; k += ofsz) 
		//for (int j = 0; j <= bsz_y+1; ++j)
		for(int j = tidy; j <= bsz_y + 1; j += ofsy) 
			//for (int i = 0; i <= bsz_x+1; ++i) 
			for(int i = tidx; i <= bsz_x + 1; i += ofsx)
				if(i*j*k == 0 || (bsz_x+1-i)*(bsz_y+1-j)*(bsz_z+1-k)==0) {
					new_data_d[ind(i, j, k, bsz_x+2, bsz_y+2)] = data_d[ind(i, j, k, bsz_x+2, bsz_y+2)];
				} else {
					new_data_d[ind(i, j, k, bsz_x+2, bsz_y+2)] =
																 ((data_d[ind(i+1, j,   k,   bsz_x+2, bsz_y+2)] 
																+  data_d[ind(i-1, j,   k,   bsz_x+2, bsz_y+2)]) *h_x 
																+ (data_d[ind(i,   j+1, k,   bsz_x+2, bsz_y+2)] 
																+  data_d[ind(i,   j-1, k,   bsz_x+2, bsz_y+2)]) *h_y
																+ (data_d[ind(i,   j,   k+1, bsz_x+2, bsz_y+2)] 
																+  data_d[ind(i,   j,   k-1, bsz_x+2, bsz_y+2)]) *h_z)
																/ (2*(h_x + h_y + h_z)); 
				}

	__syncthreads();

	//в data запиcывается разница между новым и старым значением (а в рамке -- нули)
	//for(int k = 0; k <= bsz_z+1; ++k)
	for(int k = tidz; k <= bsz_z + 1; k += ofsz)  
		//for (int j = 0; j <= bsz_y+1; ++j)
		for(int j = tidy; j <= bsz_y + 1; j += ofsy) 
			for(int i = tidx; i <= bsz_x + 1; i += ofsx) 
			//for (int i = 0; i <= bsz_x+1; ++i)
					data_d[ind(i, j, k, bsz_x+2, bsz_y+2)] -= new_data_d[ind(i, j, k, bsz_x+2, bsz_y+2)];
}

int main (int argc, char *argv[]) {
	// Initialize MPI
	MPI::Init(argc,argv);


	//double Time_work = MPI_Wtime();

	// Get the number of processes
	int n_proc_total = MPI::COMM_WORLD.Get_size();
	// Get the ID of the process
	int pid       = MPI::COMM_WORLD.Get_rank();

/*При помощи cudaGetDeviceCount(&devCount) можно получить кол-во gpu. 
В зависимости от rank' процесса выбирается одна из видеокарт с помошью
cudaSetDevice(rank % devCount). 
Когда на одной машине будет запущено несколько процессов, все видеокарты будут 
более менее равномерно загружены.
*/
	int n_gpus = 0;
	CSC(cudaGetDeviceCount(&n_gpus));
	CSC(cudaSetDevice(pid % n_gpus));

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
 	double precision, max_error_iter, max_error_block_h;
 	double lx, ly, lz;
 	double u_down, u_up, u_left, u_right, u_front, u_back;
 	double u_0;

 	if (pid==0) {
 		cin >> n_proc_x >> n_proc_y >> n_proc_z;
 		if(n_proc_x * n_proc_y * n_proc_z != n_proc_total) {
 			cerr << "Incorrect number of processes, should be: -np " << n_proc_x * n_proc_y * n_proc_z << endl;
 			MPI::COMM_WORLD.Abort(1);
 		}
	 	cin >> bsz_x >> bsz_y >> bsz_z;
	 	cin >> filename;
	 	cin >> precision >> lx >> ly >> lz;
	 	cin >> u_down >> u_up >> u_left >> u_right >> u_front >> u_back >> u_0;

	 	cerr << n_proc_x << " " << n_proc_y << " " << n_proc_z << " " << endl;
	 	cerr << bsz_x << " " << bsz_y << " " << bsz_z << endl;
	 	cerr << precision << " " << lx << " " << ly << " " << lz << endl;
	 	cerr << u_down << " " << u_up << " " << u_left << " " << u_right << " " << u_front << " " << u_back << " " << u_0 << endl;

 	} 
	
	MPI::COMM_WORLD.Bcast(&n_proc_x, 1, MPI::INT, 0);
	MPI::COMM_WORLD.Bcast(&n_proc_y, 1, MPI::INT, 0);
	MPI::COMM_WORLD.Bcast(&n_proc_z, 1, MPI::INT, 0);
	MPI::COMM_WORLD.Bcast(&bsz_x, 1, MPI::INT, 0);
	MPI::COMM_WORLD.Bcast(&bsz_y, 1, MPI::INT, 0);
	MPI::COMM_WORLD.Bcast(&bsz_z, 1, MPI::INT, 0);
	MPI::COMM_WORLD.Bcast(&precision, 1, MPI::DOUBLE, 0);
	MPI::COMM_WORLD.Bcast(&lx, 1, MPI::DOUBLE, 0);
	MPI::COMM_WORLD.Bcast(&ly, 1, MPI::DOUBLE, 0);
	MPI::COMM_WORLD.Bcast(&lz, 1, MPI::DOUBLE, 0);
	MPI::COMM_WORLD.Bcast(&u_down, 1, MPI::DOUBLE, 0);
	MPI::COMM_WORLD.Bcast(&u_up, 1, MPI::DOUBLE, 0);
	MPI::COMM_WORLD.Bcast(&u_left, 1, MPI::DOUBLE, 0);
	MPI::COMM_WORLD.Bcast(&u_right, 1, MPI::DOUBLE, 0);
	MPI::COMM_WORLD.Bcast(&u_front, 1, MPI::DOUBLE, 0);
	MPI::COMM_WORLD.Bcast(&u_back, 1, MPI::DOUBLE, 0);
	MPI::COMM_WORLD.Bcast(&u_0, 1, MPI::DOUBLE, 0);
	
	int n_x, n_y, n_z;
	n_x = n_proc_x * bsz_x;
	n_y = n_proc_y * bsz_y;
	n_z = n_proc_z * bsz_z;

	double h_x, h_y, h_z;
	h_x = pow(lx / n_x, -2);
	h_y = pow(ly / n_y, -2);
	h_z = pow(lz / n_z, -2);
	int idx = pid;
	int pz = idx / (n_proc_x * n_proc_y);
    idx -= (pz * n_proc_x * n_proc_y);
    int py = idx / n_proc_x;
    int px = idx % n_proc_x;


    int i_start = px*bsz_x, i_end = (px+1)*bsz_x+1;
    int j_start = py*bsz_y, j_end = (py+1)*bsz_y+1;
    int k_start = pz*bsz_z, k_end = (pz+1)*bsz_z+1;
 //    cout << get_pid(pid, n_proc_x, n_proc_y) 
 //    + "\tx: " + to_string(i_start) + "-" + to_string(i_end) 
	// + "\ty: " + to_string(j_start) + "-" + to_string(j_end) 
	// + "\tz: " + to_string(k_start) + "-" + to_string(k_end) + "\n";
	double *data_h, *new_data_h;
	int n_cells_in_block = (bsz_x+2)*(bsz_y+2)*(bsz_z+2);
	data_h = new double[n_cells_in_block];
	new_data_h = new double[n_cells_in_block];
	for (int i = i_start; i <= i_end; ++i){
		for (int j = j_start; j <= j_end; ++j){
			for (int k = k_start; k <= k_end; ++k){
				int i_local = i - i_start;
				int j_local = j - j_start;
				int k_local = k - k_start;

				int local1d = ind(i_local, j_local, k_local, 
									  bsz_x+2, bsz_y+2);
				//fout << to_string(i) + to_string(j) + to_string(k) + " " + to_string(i_local) + to_string(j_local) + to_string(k_local) + "_" + to_string(local1d) +  "\n" ;

				if(i == 0)
					data_h[local1d] = u_left;
				else if(i == n_x+1)
					data_h[local1d] = u_right;
				if (j == 0)
					data_h[local1d] = u_front;
				else if(j == n_y+1)
					data_h[local1d] = u_back;
				if (k == 0)
					data_h[local1d] = u_down;
				else if(k == n_z+1)
					data_h[local1d] = u_up;
				
				if(i*j*k > 0 && (n_x+1-i)*(n_y+1-j)*(n_z+1-k)>0)
					data_h[local1d] = u_0;
			}
		}
	}


	double *data_d, *new_data_d;//, *max_error_block_d;
	//cudaMalloc(&max_error_block_d, sizeof(double));
	CSC(cudaMalloc(&data_d, sizeof(double) * n_cells_in_block));
	CSC(cudaMalloc(&new_data_d, sizeof(double) * n_cells_in_block));
	CSC(cudaMemcpy(data_d, data_h, sizeof(double) * n_cells_in_block, cudaMemcpyHostToDevice));

	MPI::COMM_WORLD.Barrier();
	
	// Measure the current time
	double start = MPI::Wtime();


	int iter = 0;
	// fout << "\n-----------------------------\n" + 
	// 		printOutput(pid, n_proc_x, n_proc_y, data_h, bsz_x, bsz_y, bsz_z, iter) + 
	// 		"\n-----------------------------\n";
	//ofstream fout("out_P" + to_string(pid), std::ofstream::out);
	do {

		iter++;
		// if(iter%20 == 0 && pid == 0)
		// 	cerr << "iter " + to_string(iter) + "\n";
		// if(pid==0)
		// 	cerr << "*";
	
		//На первом этапе происходит обмен граничными слоями между процессами



		if(px > 0) {
			// cerr << "iter " + to_string(iter) + " | " + get_pid(pid, n_proc_x, n_proc_y) + "sent LTR to "  
			//       + get_pid(ind(px-1, py, pz, n_proc_x, n_proc_y), n_proc_x, n_proc_y) + "\n";
			double * border_send_h = new double[bsz_y*bsz_z];
			for (int j = 1; j <= bsz_y; ++j) 
				for (int k = 1; k <= bsz_z; ++k) 
					border_send_h[(k-1)+(j-1)*bsz_z] = data_h[ind(1, j, k, bsz_x+2, bsz_y+2)];

			MPI::COMM_WORLD.Send(border_send_h,	bsz_y*bsz_z, MPI::DOUBLE, 
				ind(px-1, py, pz, n_proc_x, n_proc_y), BorderDir::LeftToRight);
			delete[] border_send_h;
		}

		if(px < n_proc_x - 1) {
			double * border_recv_h = new double[bsz_y*bsz_z];
			// cerr << "iter " + to_string(iter) + " | " + get_pid(pid, n_proc_x, n_proc_y) + "recv LTR fr "  
			//       + get_pid(ind(px+1, py, pz, n_proc_x, n_proc_y), n_proc_x, n_proc_y) + "\n";

			MPI::COMM_WORLD.Recv(border_recv_h,	bsz_y*bsz_z, MPI::DOUBLE, 
				ind(px+1, py, pz, n_proc_x, n_proc_y), BorderDir::LeftToRight);
			for (int j = 1; j <= bsz_y; ++j) 
				for (int k = 1; k <= bsz_z; ++k) 
					data_h[ind(bsz_x+1, j, k, bsz_x+2, bsz_y+2)] = border_recv_h[(k-1)+(j-1)*bsz_z];
			delete[] border_recv_h;
		}

		//MPI::COMM_WORLD.Barrier();

		//RTL
		if (px < n_proc_x - 1) {
			// cerr << "iter " + to_string(iter) + " | " + get_pid(pid, n_proc_x, n_proc_y) + "sent RTL to "  
			//       + get_pid(ind(px+1, py, pz, n_proc_x, n_proc_y), n_proc_x, n_proc_y) + "\n";
			double * border_send_h = new double[bsz_y*bsz_z];
			for (int j = 1; j <= bsz_y; ++j) 
				for (int k = 1; k <= bsz_z; ++k) 
					border_send_h[(k-1)+(j-1)*bsz_z] = data_h[ind(bsz_x, j, k, bsz_x+2, bsz_y+2)];

			MPI::COMM_WORLD.Send(border_send_h, bsz_y*bsz_z, MPI::DOUBLE, 
				ind(px+1, py, pz, n_proc_x, n_proc_y), BorderDir::RightToLeft);
			delete[] border_send_h;
		}

		if (px > 0) {
			double * border_recv_h = new double[bsz_y*bsz_z];
			// cerr << "iter " + to_string(iter) + " | " + get_pid(pid, n_proc_x, n_proc_y) + "recv RTL fr "  
			//       + get_pid(ind(px-1, py, pz, n_proc_x, n_proc_y), n_proc_x, n_proc_y) + "\n";

			MPI::COMM_WORLD.Recv(border_recv_h, bsz_y*bsz_z, MPI::DOUBLE, 
				ind(px-1, py, pz, n_proc_x, n_proc_y), BorderDir::RightToLeft);
			for (int j = 1; j <= bsz_y; ++j) 
				for (int k = 1; k <= bsz_z; ++k) 
					data_h[ind(0, j, k, bsz_x+2, bsz_y+2)] = border_recv_h[(k-1)+(j-1)*bsz_z];

			delete[] border_recv_h;
		}
		//MPI::COMM_WORLD.Barrier();

// 		//FTB

		if(py > 0) {
			// cerr << "iter " + to_string(iter) + " | " + get_pid(pid, n_proc_x, n_proc_y) + "sent FTB to "  
			//       + get_pid(ind(px, py-1, pz, n_proc_x, n_proc_y), n_proc_x, n_proc_y) + "\n";
			double * border_send_h = new double[bsz_x*bsz_z];
			for (int k = 1; k <= bsz_z; ++k) 
				memcpy(border_send_h + (k-1)*bsz_x, data_h + ind(1, 1, k, bsz_x+2, bsz_y+2), sizeof(double)*bsz_x);

			MPI::COMM_WORLD.Send(border_send_h,	bsz_x*bsz_z, MPI::DOUBLE, 
				ind(px, py-1, pz, n_proc_x, n_proc_y), BorderDir::FrontToBack);
			delete[] border_send_h;
		}

		if(py < n_proc_y - 1){
			double * border_recv_h = new double[bsz_x*bsz_z];
			// cerr << "iter " + to_string(iter) + " | " + get_pid(pid, n_proc_x, n_proc_y) + "recv FTB fr "  
			//       + get_pid(ind(px, py+1, pz, n_proc_x, n_proc_y), n_proc_x, n_proc_y) + "\n";

			MPI::COMM_WORLD.Recv(border_recv_h,	bsz_x*bsz_z, MPI::DOUBLE, 
				ind(px, py+1, pz, n_proc_x, n_proc_y), BorderDir::FrontToBack);
			for (int k = 1; k <= bsz_z; ++k) 
				memcpy(data_h + ind(1, bsz_y+1, k, bsz_x+2, bsz_y+2),  border_recv_h + (k-1)*bsz_x, sizeof(double)*bsz_x);

			delete[] border_recv_h;
		}

		//MPI::COMM_WORLD.Barrier();

		//BTF
		
		if (py < n_proc_y - 1) {
			// cerr << "iter " + to_string(iter) + " | " + get_pid(pid, n_proc_x, n_proc_y) + "sent BTF to "  
			//       + get_pid(ind(px, py+1, pz, n_proc_x, n_proc_y), n_proc_x, n_proc_y) + "\n";
			double * border_send_h = new double[bsz_x*bsz_z];
			for (int k = 1; k <= bsz_z; ++k)
				memcpy(border_send_h + (k-1)*bsz_x, data_h + ind(1, bsz_y, k, bsz_x+2, bsz_y+2), sizeof(double)*bsz_x);

			MPI::COMM_WORLD.Send(border_send_h, bsz_x*bsz_z, MPI::DOUBLE, 
				ind(px, py+1, pz, n_proc_x, n_proc_y), BorderDir::BackToFront);
			delete[] border_send_h;
		}

		if (py > 0) {
			double * border_recv_h = new double[bsz_x*bsz_z];
			// cerr << "iter " + to_string(iter) + " | " + get_pid(pid, n_proc_x, n_proc_y) + "recv BTF fr "  
			//       + get_pid(ind(px, py-1, pz, n_proc_x, n_proc_y), n_proc_x, n_proc_y) + "\n";

			MPI::COMM_WORLD.Recv(border_recv_h, bsz_x*bsz_z, MPI::DOUBLE, 
				ind(px, py-1, pz, n_proc_x, n_proc_y), BorderDir::BackToFront);
			for (int k = 1; k <= bsz_z; ++k)
				memcpy(data_h + ind(1, 0, k, bsz_x+2, bsz_y+2), border_recv_h + (k-1)*bsz_x, sizeof(double)*bsz_x);
			delete[] border_recv_h;
		}
		//MPI::COMM_WORLD.Barrier();

// 		//UTD

		if(pz > 0) {
			// cerr << "iter " + to_string(iter) + " | " + get_pid(pid, n_proc_x, n_proc_y) + "sent UTD to "  
			//       + get_pid(ind(px, py, pz-1, n_proc_x, n_proc_y), n_proc_x, n_proc_y) + "\n";
			double * border_send_h = new double[bsz_x*bsz_y];
			for (int j = 1; j <= bsz_y; ++j)
				memcpy(border_send_h + (j-1)*bsz_x, data_h + ind(1, j, 1, bsz_x+2, bsz_y+2), sizeof(double)*bsz_x);	

			MPI::COMM_WORLD.Send(border_send_h, bsz_x*bsz_y, MPI::DOUBLE, 
				ind(px, py, pz-1, n_proc_x, n_proc_y), BorderDir::UpToDown);
			delete[] border_send_h;
		}

		if(pz < n_proc_z - 1){
			double * border_recv_h = new double[bsz_x*bsz_y];
			// cerr << "iter " + to_string(iter) + " | " + get_pid(pid, n_proc_x, n_proc_y) + "recv UTD fr "  
			//       + get_pid(ind(px, py, pz+1, n_proc_x, n_proc_y), n_proc_x, n_proc_y) + "\n";

			MPI::COMM_WORLD.Recv(border_recv_h, bsz_x*bsz_y, MPI::DOUBLE, 
				ind(px, py, pz+1, n_proc_x, n_proc_y), BorderDir::UpToDown);
			for (int j = 1; j <= bsz_y; ++j)
				memcpy(data_h + ind(1, j, bsz_z+1, bsz_x+2, bsz_y+2), border_recv_h + (j-1)*bsz_x, sizeof(double)*bsz_x);
			delete[] border_recv_h;
		}

		//MPI::COMM_WORLD.Barrier();

		//DTU

		if (pz < n_proc_z - 1) {
			// cerr << "iter " + to_string(iter) + " | " + get_pid(pid, n_proc_x, n_proc_y) + "sent DTU to "  
			//       + get_pid(ind(px, py, pz+1, n_proc_x, n_proc_y), n_proc_x, n_proc_y) + "\n";
			double * border_send_h = new double[bsz_x*bsz_y];
			for (int j = 1; j <= bsz_y; ++j)
				memcpy(border_send_h + (j-1)*bsz_x, data_h + ind(1, j, bsz_z, bsz_x+2, bsz_y+2), sizeof(double)*bsz_x);

			MPI::COMM_WORLD.Send(border_send_h, bsz_x*bsz_y, MPI::DOUBLE, 
				ind(px, py, pz+1, n_proc_x, n_proc_y), BorderDir::DownToUp);
			delete [] border_send_h;
		}

		if (pz > 0) {
			double * border_recv_h = new double[bsz_x*bsz_y];
			// cerr << "iter " + to_string(iter) + " | " + get_pid(pid, n_proc_x, n_proc_y) + "recv DTU fr "  
			//       + get_pid(ind(px, py, pz-1, n_proc_x, n_proc_y), n_proc_x, n_proc_y) + "\n";

			MPI::COMM_WORLD.Recv(border_recv_h, bsz_x*bsz_y, MPI::DOUBLE, 
				ind(px, py, pz-1, n_proc_x, n_proc_y), BorderDir::DownToUp);
			for (int j = 1; j <= bsz_y; ++j) 
				memcpy(data_h + ind(1, j, 0, bsz_x+2, bsz_y+2), border_recv_h + (j-1)*bsz_x, sizeof(double)*bsz_x);
			delete[] border_recv_h;
		}
		MPI::COMM_WORLD.Barrier();

//На втором этапе выполняется обновление значений во всех ячейках

		//double * new_data_h = new double[n_cells_in_block];
		//memcpy(new_data_h, data_h, sizeof(double)*n_cells_in_block);
		// fout << "NEW BEFORE\n" + printOutput(pid, n_proc_x, n_proc_y, data_h, bsz_x, bsz_y, bsz_z, iter);
		CSC(cudaMemcpy(data_d, data_h, sizeof(double)*n_cells_in_block, cudaMemcpyHostToDevice));
		int n_gpu_threads_x = 32, n_gpu_threads_y = 8, n_gpu_threads_z, tmp;

		cudaDeviceProp prop;
    	cudaGetDeviceProperties(&prop, pid % n_gpus);
		tmp = prop.maxThreadsPerBlock;

		//if(!pid)
		//	cerr << tmp << "\n";
		n_gpu_threads_z = tmp / (n_gpu_threads_x * n_gpu_threads_y);
		jacobi_kernel<<<1, dim3(n_gpu_threads_x, n_gpu_threads_y, n_gpu_threads_z)>>>(data_d, new_data_d, bsz_x, bsz_y, bsz_z, h_x, h_y, h_z);
		CSC(cudaGetLastError());
		// cudaMemcpy(data_h, data_d, sizeof(double)*n_cells_in_block, cudaMemcpyDeviceToHost);
		// fout << "DIFF AFTER\n" + printOutput(pid, n_proc_x, n_proc_y, data_h, bsz_x, bsz_y, bsz_z, iter);

		cudaMemcpy(data_h, new_data_d, sizeof(double)*n_cells_in_block, cudaMemcpyDeviceToHost);
		// fout << "NEW AFTER\n" + printOutput(pid, n_proc_x, n_proc_y, data_h, bsz_x, bsz_y, bsz_z, iter);

		//MPI::COMM_WORLD.Barrier();

		thrust::device_ptr<double> devPtr = thrust::device_pointer_cast(data_d);
		thrust::device_ptr<double> maxPtr = thrust::max_element(devPtr, devPtr + n_cells_in_block, cmpAbsDoubles());
		max_error_block_h = abs(*maxPtr);

		//cudaMemcpy(&max_error_block_h, max_error_block_d, sizeof(double), cudaMemcpyDeviceToHost);
		// fout << "ERROR BLOCK = " + to_string(max_error_block_h) + "\n";

		//memcpy(data_h, new_data_h, sizeof(double)*n_cells_in_block);
		//delete[] new_data_h;
		
		MPI::COMM_WORLD.Barrier();
		//fout << "data_h AFTER\n" + printOutput(pid, n_proc_x, n_proc_y, data_h, bsz_x, bsz_y, bsz_z, iter);

		// Sum the error of all the processes
		// Output is stored in the variable ’error’ of all processes
		MPI::COMM_WORLD.Allreduce(&max_error_block_h, &max_error_iter, 1, MPI::DOUBLE, MPI::MAX);
		// if(pid==0)
		// 	cerr << "i " + to_string(iter) + ": e=" + to_string(max_error_iter) + "\n";
		// if(iter==10)
		// 	break;

	
	} while(max_error_iter > precision);
	double t = MPI_Wtime() - start;
	if(!pid)
		cerr << "\n" << iter << " " << t << "\n";



	if (pid == 0) {
		ofstream resout(filename, std::ofstream::out);
		for (int pz = 0; pz < n_proc_z; ++pz) {
			for (int k = 0; k < bsz_z; ++k) {
				for (int py = 0; py < n_proc_y; ++py) {
					for (int j = 0; j < bsz_y; ++j) {
						for (int px = 0; px < n_proc_x; ++px) {
							double *temp_data = new double[bsz_x];
							if(px + py + pz == 0) {
								memcpy(temp_data, 
									data_h + ind(1, j+1, k+1, bsz_x+2, bsz_y+2), 
									bsz_x*sizeof(double));
							} else {
								// cerr << " P0 recv row [" + to_string(j) + "," + to_string(k) + "]" +  
								//         + " fr " + get_pid(ind(px, py, pz, n_proc_x, n_proc_y), n_proc_x, n_proc_y) + "\n"; 
								MPI::COMM_WORLD.Recv(temp_data, bsz_x, MPI::DOUBLE, 
													 ind(px, py, pz, n_proc_x, n_proc_y), 
													 j+k*bsz_y);
							}
							for (int i = 0; i < bsz_x; ++i) {
								resout << scientific << setprecision(6) 
								       << temp_data[i] << " ";
							}

							delete[] temp_data;
						}
						resout << "\n";
					}
				}	
				resout << "\n";
			}
		}
		resout.close();
	} else {
		for (int k = 0; k < bsz_z; ++k) {
			for (int j = 0; j < bsz_y; ++j) {
				// cerr << get_pid(pid, n_proc_x, n_proc_y) 
				// + "sent row [" + to_string(j) + "," + to_string(k) + "] to P0\n";  
				MPI::COMM_WORLD.Send(
					data_h + ind(1, j+1, k+1, bsz_x+2, bsz_y+2), 
					bsz_x, MPI::DOUBLE, 0, j+k*bsz_y);
			}
		}
	}

	delete[] data_h;
	delete[] new_data_h;

	CSC(cudaFree(data_d));
	CSC(cudaFree(new_data_d));
	// Terminate MPI
	//fout.close();
	MPI::Finalize();
}
