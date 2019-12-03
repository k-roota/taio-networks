#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <chrono>

#include "Model.h"
#include "TimeSeries.h"
#include "Tanh.h"
#include "GradientDescent.h"
#include "TimeSeriesConfig.h"
#include "NetworkConfig.h"
#include "ConfigReader.h"
#include "MatrixReader.h"

using namespace Eigen;
using namespace std;

template<typename F>
void exec_time(F fun)
{
	auto t1 = chrono::high_resolution_clock::now();
	fun();
	auto t2 = chrono::high_resolution_clock::now();
	cout << "Time : " << ((t2-t1).count() / 1000000000.0) << endl;
}

void loadWeights(Model & const model, const string & const path)
{
	MatrixXd weights = model.getWeights();
	int size = weights.rows();
	ifstream file = ifstream(path);
	if (!file.good())
	{
		throw invalid_argument("Cannot load weights from " + path);
	}
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			file >> weights(i, j);
		}
	}
	model.setWeights(weights);
	if (model.checkIsBiasUsed())
	{
		VectorXd bias = model.getBias();
		for (int i = 0; i < size; i++)
		{
			file >> bias[i];
		}
		model.setBias(bias);
	}
}

void saveWeights(const Model & const model, const string & const path)
{
	ofstream file = ofstream(path);
	if (!file.good())
	{
		throw invalid_argument("Cannot save weights to " + path);
	}
	file << model.getWeights() << endl;
	if (model.checkIsBiasUsed())
	{
		file << endl << model.getBias() << endl;
	}
}

shared_ptr<TimeSeries> loadTimeSeries(const string & const path)
{
	ifstream timeSeriesStream(path);
	if (!timeSeriesStream.good())
	{
		throw invalid_argument("Cannot read the time series file");
	}
	return shared_ptr<TimeSeries>(new TimeSeries(MatrixReader::read<double>(timeSeriesStream)));
}

shared_ptr<Model> createModel(const TimeSeries& const timeSeries, const string & const timeSeriesConfigPath, const string & const networkConfigPath)
{
	ifstream timeSeriesConfigStream(timeSeriesConfigPath);
	if (!timeSeriesConfigStream.good())
	{
		throw invalid_argument("Cannot read the time series config file");
	}
	unique_ptr<TimeSeriesConfig> timeSeriesConfig = ConfigReader::readTimeSeriesConfig(timeSeriesConfigStream);
	timeSeriesConfigStream.close();

	ifstream networkConfigStream(networkConfigPath);
	if (!networkConfigStream.good())
	{
		throw invalid_argument("Cannot read the network config file");
	}
	unique_ptr<NetworkConfig> networkConfig = ConfigReader::readNetworkConfig(networkConfigStream);
	networkConfigStream.close();

	shared_ptr<Tanh> tanh(new Tanh());
	shared_ptr<GradientDescent> gradientDescent(new GradientDescent());
	vector<vector<bool>> mask = timeSeriesConfig->getMatrix(timeSeries.seriesCount);
	shared_ptr<vector<vector<bool>>> maskPtr(new vector<vector<bool>>(move(mask)));

	return shared_ptr<Model>(new Model(tanh, networkConfig->loss, gradientDescent, timeSeries.seriesCount, timeSeriesConfig->dataPortion, maskPtr,
		networkConfig->bias, networkConfig->maxLoss, networkConfig->maxIter, networkConfig->activateInput, networkConfig->revertOnIncrease, timeSeriesConfig->timeOffset));
}

int main(int argc, char* argv[])
{
	const string TIME_SERIES_DEFAULT_PATH = "Input/time_series_in.txt";
	const string TIME_SERIES_CONFIG_PATH = "Input/ts_config.txt";
	const string NETWORK_CONFIG_PATH = "Input/config.txt";
	const string OUT_WEIGHTS_PATH = "Output/weights.txt";

	try
	{
		string timeSeriesPath = TIME_SERIES_DEFAULT_PATH;
		string inWeightsPath = "";
		string mode = "--train";

		for (int i = 1; i < argc; i++)
		{
			if (strcmp(argv[i], "--in-weights") == 0)
			{
				i += 1;
				inWeightsPath = argv[i];
			}
			else if (strcmp(argv[i], "--time-series") == 0)
			{
				i += 1;
				timeSeriesPath = argv[i];
			}
			else
			{
				mode = argv[i];
			}
		}
		
		shared_ptr<TimeSeries> timeSeries = loadTimeSeries(timeSeriesPath);
		shared_ptr<Model> model = createModel(*timeSeries, TIME_SERIES_CONFIG_PATH, NETWORK_CONFIG_PATH);
		if (!inWeightsPath.empty())
		{
			loadWeights(*model, inWeightsPath);
		}

		if (mode == "--train")
		{
			model->train(*timeSeries);
			saveWeights(*model, OUT_WEIGHTS_PATH);
		}
		else if (mode == "--predict")
		{
			VectorXd X = timeSeries->getLastWindow(model->getWindowLength());
			VectorXd Y = model->predict(X);
			cout << "X" << endl << X << endl << endl << "Y" << endl << Y << endl << endl;
		}
		else if (mode == "--evaluate")
		{
			double loss = model->evaluate(*timeSeries);
			cout << "Mean loss: " << loss << endl;
		}
		else
		{
			throw invalid_argument("Unknown argument " + mode);
		}
	}
	catch (exception e)
	{
		cout << "Error: " << e.what() << endl;
	}

	
}