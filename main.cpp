#include <iostream>
#include <vector>
#include <mpi.h>

double logisticStep(double, double);
double solveLogistic(double, double, int);
void runMPI(int argc, char** argv);

int main(int argc, char** argv) {
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
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	const int steps = 100;
	const double x0 = 0.5;
	const double r_start = 2.8;
	const double r_end = 3.8;
	const double r_step = 0.1;

	std::vector<double> all_r_values;
	if (world_rank == 0)
		for (int i = 0; i <= 10; i++)
			all_r_values.push_back(r_start + i * r_step);

	int total_tasks = (r_end - r_start) / r_step + 1.0;

	std::vector<int> sendcounts(world_size);
	std::vector<int> displs(world_size);

	int remainder = total_tasks % world_size;
	int sum = 0;
	for (int i = 0; i < world_size; i++) {
		sendcounts[i] = total_tasks / world_size;
		if (i < remainder)
			sendcounts[i]++;
		displs[i] = sum;
		sum += sendcounts[i];
	}

	int local_count = sendcounts[world_rank];
	std::vector<double> local_r(local_count);
	std::vector<double> local_results(local_count);

	MPI_Scatterv(
		world_rank == 0 ? all_r_values.data() : nullptr,
		sendcounts.data(),
		displs.data(),
		MPI_DOUBLE,
		local_r.data(),
		local_count,
		MPI_DOUBLE,
		0,
		MPI_COMM_WORLD
	);

	for (int i = 0; i < local_count; i++) {
		local_results[i] = solveLogistic(local_r[i], x0, steps);
		std::cout << "[Rank " << world_rank << "] Calculated r=" << local_r[i];
		std::cout << " Result=" << local_results[i] << std::endl;
	}
}
