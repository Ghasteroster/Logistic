#include <iostream>

double logisticStep(double, double);
double solveLogistic(double, double, int) {

int main() {
	MPI_Init(&argc, &argv);

	runMPI(argc, argv);

	MPI_Finalize();

	return 0;
}

double logisticStep(double r, double x) {
	return r * x * (1.0 - x);
}

double solveLogistic(double r, double x0, int N) {
	double x = x0;

	for (int n = 0; n < N; n++)
		x = logisticStep(r, x);
	
	return x;
}

void runMPI(int argc, char** argv) {
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
}
