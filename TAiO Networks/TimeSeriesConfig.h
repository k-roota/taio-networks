#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include "MatrixReader.h"

using namespace std;

class TimeSeriesConfig
{
public:
	TimeSeriesConfig(string matrixType, int dataPortion, int timeOffset);
	vector<vector<bool>> getMatrix(int seriesCount) const;

	string matrixType;
	int dataPortion;
	int timeOffset;
};

