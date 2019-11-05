#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <chrono>
#include <memory>

#include "Model.h"
#include "TimeSeries.h"
#include "Tanh.h"
#include "Mse.h"
#include "GradientDescent.h"

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

void loadMatrix(MatrixXd & const m, const string & const path)
{
	int rows = m.rows();
	int cols = m.cols();
	ifstream file = ifstream(path);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			file >> m(i, j);
		}
	}
	file.close();
}

void saveMatrix(const MatrixXd & const m, const string & const path)
{
	ofstream file = ofstream(path);
	file << m;
	file.close();
}

VectorXd getSeriesVector(TimeSeries* series, int seriesCount, int windowLength)
{
	VectorXd X = VectorXd(seriesCount * windowLength);
	for (int i = 0; i < seriesCount; i++)
	{
		vector<double> window = series[i].getWindow(windowLength);
		for (int j = 0; j < windowLength; j++)
		{
			X[i * windowLength + j] = window[j];
		}
	}
	return X;
}

int main()
{
	//MatrixXd a = MatrixXd(1000, 1000);
	//loadMatrix(a, "C:/Users/4Ruta/OneDrive/Pulpit/Pycharm/data.txt");
	//MatrixXd b = MatrixXd(1000, 1000);
	//loadMatrix(b, "C:/Users/4Ruta/OneDrive/Pulpit/Pycharm/data2.txt");
	//
	//auto t1 = chrono::high_resolution_clock::now();

	//MatrixXd c;

	//for (int i = 0; i < 1; i++)
	//{
	//	c = a * b;
	//	c = b * a;
	//	/*return c_val;*/
	//}

	//

	//auto t2 = chrono::high_resolution_clock::now();
	//cout << "Time : " << ((t2 - t1).count() / 1000000000.0) << endl;

	////saveMatrix(*c, "C:/Users/4Ruta/OneDrive/Pulpit/Pycharm/EigenResult.txt");

	/*int n = 10 * 10;
	bool** mask = new bool*[n];
	for (int i = 0; i < n; i++)
	{
		mask[i] = new bool[n];
		for (int j = 0; j < n; j++)
		{
			mask[i][j] = (i > 10*j);
		}
	}*/

	const int seriesCount = 2;
	const int seriesLength = 10;
	const int windowLength = 3;
	const int epochs = 100;

	//double data[seriesCount][seriesLength] = { { 2, 4, 8, 16, 32, 64, 128 }, {3, 6, 12, 24} };
	double max = (seriesCount + 1) * pow(2, seriesLength);
	TimeSeries *series = new TimeSeries[seriesCount];
	for (int i = 0; i < seriesCount; i++)
	{
		for (int j = 0; j < seriesLength; j++)
		{
			series[i].add((i + 2) * pow(2, j) / max);
		}
	}

	//for (double tau = 1; tau < 20; tau += 1)
	//{
		Tanh tanh = Tanh();
		Mse mse = Mse();
		GradientDescent gradientDescent = GradientDescent();

		Model model = Model(&tanh, &mse, &gradientDescent, seriesCount, windowLength);

		for (int e = 0; e < epochs; e++)
		{
			VectorXd Y = getSeriesVector(series, seriesCount, windowLength);
			for (int i = 0; i < seriesLength - windowLength; i++)
			{
				VectorXd X = Y;
				Y = getSeriesVector(series, seriesCount, windowLength);
				//cout << endl << "X" << endl << X << endl << endl << "Y" << endl << Y << endl << endl;
				model.trainStep(X, Y);
			}
			for (int i = 0; i < seriesCount; i++)
			{
				series[i].reset();
			}
			cout << endl;
		}

		double loss = 0.0;
		VectorXd X;
		VectorXd Y = getSeriesVector(series, seriesCount, windowLength);
		for (int i = 0; i < seriesLength - windowLength; i++)
		{
			X = Y;
			Y = getSeriesVector(series, seriesCount, windowLength);
			//cout << endl << "X" << endl << X << endl << endl << "Y" << endl << Y << endl << endl;
			double newLoss = mse(model.predict(X), Y);
			if (newLoss > loss)
				loss = newLoss;
		}
		for (int i = 0; i < seriesCount; i++)
		{
			series[i].reset();
		}

		cout << "Loss : " << loss << endl;

		cout << "Predict:" << endl << Y << endl << endl << model.predict(Y) << endl << endl;

		cout << model.getWeights();
	//}

	

}