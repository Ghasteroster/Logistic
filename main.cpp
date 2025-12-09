#include <iostream>

double logisticStep(double, double);
double solveLogistic(double, double, int) {

int main() {

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
