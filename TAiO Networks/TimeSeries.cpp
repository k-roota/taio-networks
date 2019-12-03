#include "TimeSeries.h"

TimeSeries::TimeSeries(vector<vector<double>> matrix):
	matrix(matrix),
	seriesLength(matrix.size()),
	seriesCount(matrix[0].size())
{}

VectorXd TimeSeries::getWindow(int id, int length) const
{
	VectorXd window(seriesCount * length);

	for (int i = 0; i < length; i++)
	{
		for (int j = 0; j < seriesCount; j++)
		{
			window[i * seriesCount + j] = matrix[i + id][j];
		}
	}

	return window;
}

int TimeSeries::getLastId(int windowLength) const
{
	return seriesLength - windowLength;
}

VectorXd TimeSeries::getLastWindow(int length) const
{
	return getWindow(getLastId(length), length);
}
