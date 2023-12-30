#include <stdlib.h>
#include <iostream>
#include <string>
#include "mpi.h"
#include <iomanip>
#include <fstream>

int main (int argc, char *argv[]) {
	
    int rank;
    // Initialize MPI
    MPI_Status status;
	MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int a[5];
    a[0] = 10;
    a[1] = rank;
    if (rank == 1)
        MPI_Send(&a[1], 1, MPI_INT, 0, 99, MPI_COMM_WORLD);
    else {
        MPI_Recv(&a[0], 1, MPI_INT, 1, 99, MPI_COMM_WORLD, &status);
        std::cout << a[0];
    }

    MPI_Finalize();
    return 0;
}
