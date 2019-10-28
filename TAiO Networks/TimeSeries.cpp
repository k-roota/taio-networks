#include "TimeSeries.h"

TimeSeries::TimeSeries()
	:newStartId(0)
{
}

TimeSeries::TimeSeries(double* start, double* end)
	: series(start, end), newStartId(0)
{
}

VectorXd TimeSeries::getWindow(int size)
{
	VectorXd window = VectorXd(size);
	for (int i = 0; i < size; i++)
	{
		window[i] = series[i + newStartId];
	}
	newStartId++;
	return window;
}

void TimeSeries::add(double* start, double* end)
{
	series.insert(series.end(), start, end);
}

