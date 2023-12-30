#include <stdlib.h>
#include <iostream>
#include <string>
#include <cstring>
#include "mpi.h"
#include <iomanip>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

enum BorderDir {
	LeftToRight, RightToLeft,
	UpToDown, DownToUp,
	FrontToBack, BackToFront,
	Out,
};

int ind(int i, int j, int k, int wx, int wy) {
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

// string printOutput(int pid, int n_proc_x, int n_proc_y, double *data, int blocksize_x, int blocksize_y, int blocksize_z, int iter) {
// 	//ofstream fout(fn, std::ofstream::out);
// 	ostringstream oss;
// 	oss << get_pid(pid, n_proc_y, n_proc_y) + ": " + to_string(iter) << endl;
// 	//fout << fn << endl;
// 	for (int k = 0; k < blocksize_z+2; ++k){
// 		for (int j = 0; j < blocksize_y+2; ++j) {
// 			for (int i = 0; i < blocksize_x+2; ++i) {
// 				oss << scientific << setprecision(6) << data[ind(i, j, k, blocksize_x+2, blocksize_y+2)] << "\t";
// 			}
// 			oss << "\n";
// 		}
// 		oss << "\n==========================\n";
// 	}
// 	return oss.str();
// }

// void print(int blocksize_x, int blocksize_y, int blocksize_z, double *data, ofstream &resout) {
// 	for (int k = 1; k <= blocksize_z; ++k) {
// 			for(int j = 1; j <= blocksize_y; ++j) {
// 				for (int i = 1; i <= blocksize_x; ++i) {
// 					resout << scientific << setprecision(6) << data[ind(i, j, k, blocksize_x+2, blocksize_y+2)] << " "; 
// 				}
// 				resout << "\n";
// 			}
// 			resout << "\n";
// 		}
// }



int main (int argc, char *argv[]) {
	// Initialize MPI
	MPI::Init(argc,argv);

	//double Time_work = MPI_Wtime();

	// Get the number of processes
	int n_proc_total = MPI::COMM_WORLD.Get_size();
	// Get the ID of the process
	int pid       = MPI::COMM_WORLD.Get_rank();

// На первой строке заданы три числа: размер сетки
// процессов. Гарантируется, что при запуске программы количество процессов будет
// равно произведению этих трех чисел. На второй строке задается размер блока,
// который будет обрабатываться одним процессом: три числа. Далее задается путь к
// выходному файлу, в который необходимо записать конечный результат работы
// программы и точность ε . На последующих строках описывается задача: задаются
// размеры области lx , ly и lz , граничные условия: udown
// , uup , uleft , uright , ufront и uback , и начальное значение u .
 
 	int n_proc_x, n_proc_y, n_proc_z;
 	int blocksize_x, blocksize_y, blocksize_z;
 	string filename;
 	double precision, max_error_iter;
 	double lx, ly, lz;
 	double u_down, u_up, u_left, u_right, u_front, u_back;
 	double u_0;

 	if (pid==0) {
 		cin >> n_proc_x >> n_proc_y >> n_proc_z;
 		if(n_proc_x * n_proc_y * n_proc_z != n_proc_total) {
 			cerr << "Incorrect number of processes, should be: -np " << n_proc_x * n_proc_y * n_proc_z << endl;
 			MPI::COMM_WORLD.Abort(1);
 		}
	 	cin >> blocksize_x >> blocksize_y >> blocksize_z;
	 	cin >> filename;
	 	cin >> precision >> lx >> ly >> lz;
	 	cin >> u_down >> u_up >> u_left >> u_right >> u_front >> u_back >> u_0;

	 	cerr << n_proc_x << " " << n_proc_y << " " << n_proc_z << " " << endl;
	 	cerr << blocksize_x << " " << blocksize_y << " " << blocksize_z << endl;
	 	cerr << precision << " " << lx << " " << ly << " " << lz << endl;
	 	cerr << u_down << " " << u_up << " " << u_left << " " << u_right << " " << u_front << " " << u_back << " " << u_0 << endl;

 	} 
	
	MPI::COMM_WORLD.Bcast(&n_proc_x, 1, MPI::INT, 0);
	MPI::COMM_WORLD.Bcast(&n_proc_y, 1, MPI::INT, 0);
	MPI::COMM_WORLD.Bcast(&n_proc_z, 1, MPI::INT, 0);
	MPI::COMM_WORLD.Bcast(&blocksize_x, 1, MPI::INT, 0);
	MPI::COMM_WORLD.Bcast(&blocksize_y, 1, MPI::INT, 0);
	MPI::COMM_WORLD.Bcast(&blocksize_z, 1, MPI::INT, 0);
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
	n_x = n_proc_x * blocksize_x;
	n_y = n_proc_y * blocksize_y;
	n_z = n_proc_z * blocksize_z;

	double h_x, h_y, h_z;
	h_x = pow(lx / n_x, -2);
	h_y = pow(ly / n_y, -2);
	h_z = pow(lz / n_z, -2);
	int idx = pid;
	int pz = idx / (n_proc_x * n_proc_y);
    idx -= (pz * n_proc_x * n_proc_y);
    int py = idx / n_proc_x;
    int px = idx % n_proc_x;


    int i_start = px*blocksize_x, i_end = (px+1)*blocksize_x+1;
    int j_start = py*blocksize_y, j_end = (py+1)*blocksize_y+1;
    int k_start = pz*blocksize_z, k_end = (pz+1)*blocksize_z+1;
 //    cout << get_pid(pid, n_proc_x, n_proc_y) 
 //    + "\tx: " + to_string(i_start) + "-" + to_string(i_end) 
	// + "\ty: " + to_string(j_start) + "-" + to_string(j_end) 
	// + "\tz: " + to_string(k_start) + "-" + to_string(k_end) + "\n";
	double *data;
	int n_cells_in_block = (blocksize_x+2)*(blocksize_y+2)*(blocksize_z+2);

	//ofstream fout("out_P" + to_string(pid), std::ofstream::out);

	data = new double[n_cells_in_block];
	for (int i = i_start; i <= i_end; ++i){
		for (int j = j_start; j <= j_end; ++j){
			for (int k = k_start; k <= k_end; ++k){
				int i_local = i - i_start;
				int j_local = j - j_start;
				int k_local = k - k_start;

				int local1d = ind(i_local, j_local, k_local, 
									  blocksize_x+2, blocksize_y+2);
				//fout << to_string(i) + to_string(j) + to_string(k) + " " + to_string(i_local) + to_string(j_local) + to_string(k_local) + "_" + to_string(local1d) +  "\n" ;

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

	MPI::COMM_WORLD.Barrier();
	
	// Measure the current time
	//double start = MPI::Wtime();


	int iter = 0;
	// fout << "\n-----------------------------\n" + 
	// 		printOutput(pid, n_proc_x, n_proc_y, data, blocksize_x, blocksize_y, blocksize_z, iter) + 
	// 		"\n-----------------------------\n";

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
			MPI::COMM_WORLD.Send(data, n_cells_in_block, MPI::DOUBLE, 
				ind(px-1, py, pz, n_proc_x, n_proc_y), BorderDir::LeftToRight);
		}

		if(px < n_proc_x - 1) {
			double * border_data = new double[n_cells_in_block];
			// cerr << "iter " + to_string(iter) + " | " + get_pid(pid, n_proc_x, n_proc_y) + "recv LTR fr "  
			//       + get_pid(ind(px+1, py, pz, n_proc_x, n_proc_y), n_proc_x, n_proc_y) + "\n";

			MPI::COMM_WORLD.Recv(border_data, n_cells_in_block, MPI::DOUBLE, 
				ind(px+1, py, pz, n_proc_x, n_proc_y), BorderDir::LeftToRight);
			for (int j = 1; j <= blocksize_y; ++j) {
				for (int k = 1; k <= blocksize_z; ++k) {
					data[ind(blocksize_x+1, j, k, blocksize_x+2, blocksize_y+2)] = 
						border_data[ind(1, j, k, blocksize_x+2, blocksize_y+2)];
				}
			}
			delete[] border_data;
		}

		MPI::COMM_WORLD.Barrier();

//copy
		//RTL
		if (px < n_proc_x - 1) {
			// cerr << "iter " + to_string(iter) + " | " + get_pid(pid, n_proc_x, n_proc_y) + "sent RTL to "  
			//       + get_pid(ind(px+1, py, pz, n_proc_x, n_proc_y), n_proc_x, n_proc_y) + "\n";
			MPI::COMM_WORLD.Send(data, n_cells_in_block, MPI::DOUBLE, 
				ind(px+1, py, pz, n_proc_x, n_proc_y), BorderDir::RightToLeft);
		}

		if (px > 0) {
			double * border_data = new double[n_cells_in_block];
			// cerr << "iter " + to_string(iter) + " | " + get_pid(pid, n_proc_x, n_proc_y) + "recv RTL fr "  
			//       + get_pid(ind(px-1, py, pz, n_proc_x, n_proc_y), n_proc_x, n_proc_y) + "\n";

			MPI::COMM_WORLD.Recv(border_data, n_cells_in_block, MPI::DOUBLE, 
				ind(px-1, py, pz, n_proc_x, n_proc_y), BorderDir::RightToLeft);
			for (int j = 1; j <= blocksize_y; ++j) {
				for (int k = 1; k <= blocksize_z; ++k) {
					data[ind(0, j, k, blocksize_x+2, blocksize_y+2)] = 
						border_data[ind(blocksize_x, j, k, blocksize_x+2, blocksize_y+2)];
				}
			}
			delete[] border_data;
		}
		MPI::COMM_WORLD.Barrier();

		//FTB

		if(py > 0) {
			// cerr << "iter " + to_string(iter) + " | " + get_pid(pid, n_proc_x, n_proc_y) + "sent FTB to "  
			//       + get_pid(ind(px, py-1, pz, n_proc_x, n_proc_y), n_proc_x, n_proc_y) + "\n";
			MPI::COMM_WORLD.Send(data, n_cells_in_block, MPI::DOUBLE, 
				ind(px, py-1, pz, n_proc_x, n_proc_y), BorderDir::FrontToBack);
		}

		if(py < n_proc_y - 1){
			double * border_data = new double[n_cells_in_block];
			// cerr << "iter " + to_string(iter) + " | " + get_pid(pid, n_proc_x, n_proc_y) + "recv FTB fr "  
			//       + get_pid(ind(px, py+1, pz, n_proc_x, n_proc_y), n_proc_x, n_proc_y) + "\n";

			MPI::COMM_WORLD.Recv(border_data, n_cells_in_block, MPI::DOUBLE, 
				ind(px, py+1, pz, n_proc_x, n_proc_y), BorderDir::FrontToBack);
			for (int i = 1; i <= blocksize_x; ++i) {
				for (int k = 1; k <= blocksize_z; ++k) {
					data[ind(i, blocksize_y+1, k, blocksize_x+2, blocksize_y+2)] = 
						border_data[ind(i, 1, k, blocksize_x+2, blocksize_y+2)];
				}
			}
			delete[] border_data;
		}

		MPI::COMM_WORLD.Barrier();

		//BTF
		
		if (py < n_proc_y - 1) {
			// cerr << "iter " + to_string(iter) + " | " + get_pid(pid, n_proc_x, n_proc_y) + "sent BTF to "  
			//       + get_pid(ind(px, py+1, pz, n_proc_x, n_proc_y), n_proc_x, n_proc_y) + "\n";
			MPI::COMM_WORLD.Send(data, n_cells_in_block, MPI::DOUBLE, 
				ind(px, py+1, pz, n_proc_x, n_proc_y), BorderDir::BackToFront);
		}

		if (py > 0) {
			double * border_data = new double[n_cells_in_block];
			// cerr << "iter " + to_string(iter) + " | " + get_pid(pid, n_proc_x, n_proc_y) + "recv BTF fr "  
			//       + get_pid(ind(px, py-1, pz, n_proc_x, n_proc_y), n_proc_x, n_proc_y) + "\n";

			MPI::COMM_WORLD.Recv(border_data, n_cells_in_block, MPI::DOUBLE, 
				ind(px, py-1, pz, n_proc_x, n_proc_y), BorderDir::BackToFront);
			for (int i = 1; i <= blocksize_x; ++i) {
				for (int k = 1; k <= blocksize_z; ++k) {
					data[ind(i, 0, k, blocksize_x+2, blocksize_y+2)] = 
						border_data[ind(i, blocksize_y, k, blocksize_x+2, blocksize_y+2)];
				}
			}
			delete[] border_data;
		}
		MPI::COMM_WORLD.Barrier();

		//UTD


		if(pz > 0) {
			// cerr << "iter " + to_string(iter) + " | " + get_pid(pid, n_proc_x, n_proc_y) + "sent UTD to "  
			//       + get_pid(ind(px, py, pz-1, n_proc_x, n_proc_y), n_proc_x, n_proc_y) + "\n";
			MPI::COMM_WORLD.Send(data, n_cells_in_block, MPI::DOUBLE, 
				ind(px, py, pz-1, n_proc_x, n_proc_y), BorderDir::UpToDown);
		}

		if(pz < n_proc_z - 1){
			double * border_data = new double[n_cells_in_block];
			// cerr << "iter " + to_string(iter) + " | " + get_pid(pid, n_proc_x, n_proc_y) + "recv UTD fr "  
			//       + get_pid(ind(px, py, pz+1, n_proc_x, n_proc_y), n_proc_x, n_proc_y) + "\n";

			MPI::COMM_WORLD.Recv(border_data, n_cells_in_block, MPI::DOUBLE, 
				ind(px, py, pz+1, n_proc_x, n_proc_y), BorderDir::UpToDown);
			for (int i = 1; i <= blocksize_x; ++i) {
				for (int j = 1; j <= blocksize_y; ++j) {
					data[ind(i, j, blocksize_z+1, blocksize_x+2, blocksize_y+2)] = 
						border_data[ind(i, j, 1, blocksize_x+2, blocksize_y+2)];
				}
			}
			delete[] border_data;
		}

		MPI::COMM_WORLD.Barrier();

		//DTU

		if (pz < n_proc_z - 1) {
			// cerr << "iter " + to_string(iter) + " | " + get_pid(pid, n_proc_x, n_proc_y) + "sent DTU to "  
			//       + get_pid(ind(px, py, pz+1, n_proc_x, n_proc_y), n_proc_x, n_proc_y) + "\n";
			MPI::COMM_WORLD.Send(data, n_cells_in_block, MPI::DOUBLE, 
				ind(px, py, pz+1, n_proc_x, n_proc_y), BorderDir::DownToUp);
		}

		if (pz > 0) {
			double * border_data = new double[n_cells_in_block];
			// cerr << "iter " + to_string(iter) + " | " + get_pid(pid, n_proc_x, n_proc_y) + "recv DTU fr "  
			//       + get_pid(ind(px, py, pz-1, n_proc_x, n_proc_y), n_proc_x, n_proc_y) + "\n";

			MPI::COMM_WORLD.Recv(border_data, n_cells_in_block, MPI::DOUBLE, 
				ind(px, py, pz-1, n_proc_x, n_proc_y), BorderDir::DownToUp);
			for (int i = 1; i <= blocksize_x; ++i) {
				for (int j = 1; j <= blocksize_y; ++j) {
					data[ind(i, j, 0, blocksize_x+2, blocksize_y+2)] = 
						border_data[ind(i, j, blocksize_z, blocksize_x+2, blocksize_y+2)];
				}
			}
			delete[] border_data;
		}
		MPI::COMM_WORLD.Barrier();

//На втором этапе выполняется обновление значений во всех ячейках

		double * new_data = new double[n_cells_in_block];
		memcpy(new_data, data, sizeof(double)*n_cells_in_block);
		//fout << "NEW BEFORE\n" + printOutput(pid, n_proc_x, n_proc_y, new_data, blocksize_x, blocksize_y, blocksize_z, iter);

		double max_error_block = 0;
		for (int i = 1; i <= blocksize_x; ++i) {
			for (int j = 1; j <= blocksize_y; ++j) {
				for(int k = 1; k <= blocksize_z; ++k) {
					new_data[ind(i, j, k, blocksize_x+2, blocksize_y+2)] =
								 ((data[ind(i+1, j,   k,   blocksize_x+2, blocksize_y+2)] 
								+  data[ind(i-1, j,   k,   blocksize_x+2, blocksize_y+2)]) *h_x 
								+ (data[ind(i,   j+1, k,   blocksize_x+2, blocksize_y+2)] 
								+  data[ind(i,   j-1, k,   blocksize_x+2, blocksize_y+2)]) *h_y
								+ (data[ind(i,   j,   k+1, blocksize_x+2, blocksize_y+2)] 
								+  data[ind(i,   j,   k-1, blocksize_x+2, blocksize_y+2)]) *h_z)
								/ (2*(h_x + h_y + h_z)); 
					double error = abs(new_data[ind(i, j, k, blocksize_x+2, blocksize_y+2)] -
						            data[ind(i, j, k, blocksize_x+2, blocksize_y+2)] );
					if(error > max_error_block)
						max_error_block = error;
				}
			}
		}


		//fout << "NEW AFTER\n" + printOutput(pid, n_proc_x, n_proc_y, new_data, blocksize_x, blocksize_y, blocksize_z, iter);

		memcpy(data, new_data, sizeof(double)*n_cells_in_block);
		delete[] new_data;
		
		MPI::COMM_WORLD.Barrier();
		//fout << "DATA AFTER\n" + printOutput(pid, n_proc_x, n_proc_y, data, blocksize_x, blocksize_y, blocksize_z, iter);

		// Sum the error of all the processes
		// Output is stored in the variable ’error’ of all processes
		MPI::COMM_WORLD.Allreduce(&max_error_block, &max_error_iter, 1, MPI::DOUBLE, MPI::MAX);
		if(pid==0)
			cerr << "i " + to_string(iter) + ": e=" + to_string(max_error_iter) + "\t";
	
	} while(max_error_iter > precision);




	if (pid == 0) {
		ofstream resout(filename, std::ofstream::out);
		for (int pz = 0; pz < n_proc_z; ++pz) {
			for (int k = 0; k < blocksize_z; ++k) {
				for (int py = 0; py < n_proc_y; ++py) {
					for (int j = 0; j < blocksize_y; ++j) {
						for (int px = 0; px < n_proc_x; ++px) {
							double *temp_data = new double[blocksize_x];
							if(px + py + pz == 0) {
								memcpy(temp_data, 
									data + ind(1, j+1, k+1, blocksize_x+2, blocksize_y+2), 
									blocksize_x*sizeof(double));
							} else {
								// cerr << " P0 recv row [" + to_string(j) + "," + to_string(k) + "]" +  
								//         + " fr " + get_pid(ind(px, py, pz, n_proc_x, n_proc_y), n_proc_x, n_proc_y) + "\n"; 
								MPI::COMM_WORLD.Recv(temp_data, blocksize_x, MPI::DOUBLE, 
													 ind(px, py, pz, n_proc_x, n_proc_y), 
													 j+k*blocksize_y);
							}
							for (int i = 0; i < blocksize_x; ++i) {
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
		for (int k = 0; k < blocksize_z; ++k) {
			for (int j = 0; j < blocksize_y; ++j) {
				// cerr << get_pid(pid, n_proc_x, n_proc_y) 
				// + "sent row [" + to_string(j) + "," + to_string(k) + "] to P0\n";  
				MPI::COMM_WORLD.Send(
					data + ind(1, j+1, k+1, blocksize_x+2, blocksize_y+2), 
					blocksize_x, MPI::DOUBLE, 0, j+k*blocksize_y);
			}
		}
	}

	//Time_work = MPI_Wtime() - Time_work;

	// Terminate MPI
	//fout.close();
	MPI::Finalize();
}
