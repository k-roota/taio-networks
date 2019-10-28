#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <chrono>

#include"Model.h"
#include"TimeSeries.h"

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
	const int seriesLength = 4;
	const int windowLength = 3;

	double data[seriesCount][seriesLength] = { { 2, 4, 8, 16 }, {3, 6, 12, 24} };
	TimeSeries *series = new TimeSeries[seriesCount];
	for (int i = 0; i < seriesCount; i++)
	{
		series[i].add(begin(data[i]), end(data[i]));
	}

	Model model = Model(seriesCount, windowLength);

	
}