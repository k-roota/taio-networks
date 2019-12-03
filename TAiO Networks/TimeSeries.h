#pragma once

#include<vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class TimeSeries
{
private:
	vector<vector<double>> matrix;

public:
	const int seriesCount;
	const int seriesLength;
	
	TimeSeries(vector<vector<double>> matrix);

	VectorXd getWindow(int id, int length) const;
	int getLastId(int windowLength) const;
	VectorXd getLastWindow(int length) const;
};

