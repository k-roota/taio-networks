#include "TimeSeries.h"

TimeSeries::TimeSeries()
	:newStartId(0)
{
}

TimeSeries::TimeSeries(double* start, double* end)
	: series(start, end), newStartId(0)
{
}

vector<double> TimeSeries::getWindow(int size)
{
	vector<double> v = vector<double>(series.begin() + newStartId, series.begin() + newStartId + size);
	newStartId++;
	return v;
}

void TimeSeries::add(double* start, double* end)
{
	series.insert(series.end(), start, end);
}

void TimeSeries::add(double x)
{
	series.push_back(x);
}

void TimeSeries::reset()
{
	newStartId = 0;
}

