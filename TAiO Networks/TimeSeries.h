#pragma once
#include<vector>
//#include<Eigen/Dense>

using namespace std;
//using namespace Eigen;

class TimeSeries
{
private:
	vector<double> series;
	int newStartId;

public:
	TimeSeries();
	TimeSeries(double *start, double *end);

	vector<double> getWindow(int size);
	void add(double* start, double* end);
	void add(double x);
	void reset();
};

