#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <mpi.h>
#include "gtest/gtest.h"

double logisticStep(double, double);
double solveLogistic(double, double, int);
void runMPI(int argc, char** argv);


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
		std::cout << "[Ранг " << world_rank << "] Вычисленное r=" << local_r[i];
		std::cout << " Результат=" << local_results[i] << std::endl;
	}

	std::vector<double> all_results;
	if (world_rank == 0)
		all_results.resize(total_tasks);
	
	MPI_Gatherv(
		local_results.data(),
		local_count,
		MPI_DOUBLE,
		world_rank == 0 ? all_results.data() : nullptr,
		sendcounts.data(),
		displs.data(),
		MPI_DOUBLE,
		0,
		MPI_COMM_WORLD
	);

	if (world_rank == 0) {
		std::string filename = "result.dat";
		std::ofstream outFile(filename);
		if (!outFile.is_open()) {
			std::cerr << "Ошибка отрытия файла для чтения!" << std::endl;
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		
		std::cout << "Вычисление завершено. Запись результатов в " << filename << ".." << std::endl;
		outFile << "Значение r\tРезультат_X100\n";
		for (int i = 0; i < total_tasks; i++)
			outFile << std::fixed << std::setprecision(1) << all_r_values[i] << "\t\t" << std::setprecision(6) << all_results[i] << "\n";

		outFile.close();
		std::cout << "Конец." << std::endl;
	}

	//MPI_Barrier(MPI_COMM_WORLD);
}
// 1) Тест с одним процессом (np 1)
TEST(MpiTest, SingleProcess) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const double r_start = 2.8;
    const double r_end = 3.8;
    const double r_step = 0.1;

    int total_tasks = (r_end - r_start) / r_step + 1.0;

    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);

    int remainder = total_tasks % size;
    int sum = 0;

    for (int i = 0; i < size; i++) {
        sendcounts[i] = total_tasks / size;
        if (i < remainder) sendcounts[i]++;
        displs[i] = sum;
        sum += sendcounts[i];
    }

    if (size == 1) {
        EXPECT_EQ(sendcounts[0], total_tasks);
        EXPECT_EQ(displs[0], 0);
    }
}

// 2) Тест на большое число процессов
TEST(MpiTest, ManyProcessesMoreThanTasks) {
    const double r_start = 2.8;
    const double r_end = 3.8;
    const double r_step = 0.1;

    int total_tasks = (r_end - r_start) / r_step + 1.0;

    int fake_size = total_tasks * 2;

    std::vector<int> sendcounts(fake_size);
    std::vector<int> displs(fake_size);

    int remainder = total_tasks % fake_size;
    int sum = 0;

    for (int i = 0; i < fake_size; i++) {
        sendcounts[i] = total_tasks / fake_size;
        if (i < remainder) sendcounts[i]++;
        displs[i] = sum;
        sum += sendcounts[i];
    }

    EXPECT_EQ(sum, total_tasks);

    bool has_zero_tasks = false;
    for (int sc : sendcounts) {
        if (sc == 0) has_zero_tasks = true;
    }

    EXPECT_TRUE(has_zero_tasks);
}

// 3) Тест с нецелым количеством задач
TEST(MpiTest, NonIntegerTaskCountDistribution) {
    double r_start = 2.8;
    double r_end = 3.77;
    double r_step = 0.1;

    int total_tasks = (r_end - r_start) / r_step + 1;

    for (int test_size : {2, 3, 4, 5}) {

        std::vector<int> sendcounts(test_size);
        std::vector<int> displs(test_size);

        int remainder = total_tasks % test_size;
        int sum = 0;

        for (int i = 0; i < test_size; i++) {
            sendcounts[i] = total_tasks / test_size;
            if (i < remainder) sendcounts[i]++;
            displs[i] = sum;
            sum += sendcounts[i];
        }

        EXPECT_EQ(sum, total_tasks);
        EXPECT_LE(std::abs(sendcounts[0] - sendcounts[test_size - 1]), 1);
    }
}

// 4) Тест граничных значений
TEST(LogisticTest, BoundaryValues) {
    EXPECT_DOUBLE_EQ(solveLogistic(0.0, 0.5, 100), 0.0);
    EXPECT_DOUBLE_EQ(solveLogistic(2.8, 0.0, 100), 0.0);
    EXPECT_DOUBLE_EQ(solveLogistic(2.8, 1.0, 1), 0.0);

    double result = solveLogistic(3.8, 0.5, 100);
    EXPECT_GE(result, 0.0);
    EXPECT_LE(result, 1.0);
}

// 5) Тест стабильности
TEST(LogisticTest, StabilityAndNoNaN) {
    double r = 3.5, x0 = 0.2;

    double last = solveLogistic(r, x0, 200);
    double repeat = solveLogistic(r, last, 200);

    EXPECT_NEAR(last, repeat, 1e-6);

    EXPECT_FALSE(std::isnan(last));
    EXPECT_FALSE(std::isinf(last));

    double chaotic = solveLogistic(4.1, 0.5, 50);
    EXPECT_FALSE(std::isnan(chaotic));
}

// 6) Тест формата файла
TEST(FileTest, OutputFormat) {
    std::vector<double> r_values = { 2.8, 2.9, 3.0 };
    std::vector<double> results = { 0.642857, 0.655172, 0.666667 };

    std::stringstream ss;
    ss << "Значение r\tРезультат_X100\n";
    for (size_t i = 0; i < r_values.size(); i++) {
        ss << std::fixed << std::setprecision(1) << r_values[i]
            << "\t\t"
            << std::setprecision(6) << results[i] << "\n";
    }

    std::string output = ss.str();

    EXPECT_TRUE(output.find("Значение r") != std::string::npos);
    EXPECT_TRUE(output.find("Результат_X100") != std::string::npos);
    EXPECT_TRUE(output.find("2.8") != std::string::npos);
    EXPECT_TRUE(output.find("0.642857") != std::string::npos);

    size_t pos = output.find("2.8");
    EXPECT_EQ(output[pos + 3], '\t');
}

// 7) Тест корректности распределения задач
TEST(MpiTest, TaskDistributionCorrectness) {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int total_tasks = 11;

    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);

    int remainder = total_tasks % size;
    int sum = 0;

    for (int i = 0; i < size; i++) {
        sendcounts[i] = total_tasks / size;
        if (i < remainder) sendcounts[i]++;
        displs[i] = sum;
        sum += sendcounts[i];
    }

    EXPECT_EQ(sum, total_tasks);
    EXPECT_LE(std::abs(sendcounts[0] - sendcounts[size - 1]), 1);

    for (int i = 1; i < size; i++) {
        EXPECT_GT(displs[i], displs[i - 1]);
    }
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // если есть аргумент --test → запускаем GTest
    bool run_tests = false;
    if (argc > 1 && std::string(argv[1]) == "--test") {
        run_tests = true;
    }

    if (run_tests) {
        ::testing::InitGoogleTest(&argc, argv);
        ::testing::FLAGS_gtest_death_test_style = "threadsafe";
        int result = RUN_ALL_TESTS();
        MPI_Finalize();
        return result;
    }

    // иначе выполняем MPI программу
    runMPI(argc, argv);

    MPI_Finalize();
    return 0;
}
