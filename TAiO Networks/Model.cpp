#include "Model.h"

#include<chrono>
#include<iostream>

using namespace std;

MatrixXd Model::initMask(bool** mask) const
{
	int size = getSize();
	if (mask == nullptr)
	{
		return MatrixXd::Constant(size, size, 1);
	}
	else
	{
		MatrixXd numMask(size, size);
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < size; j++)
			{
				mask[i, j] ? numMask(i, j) = 1 : numMask(i, j) = 0;
			}
		}
		return numMask;
	}
}

MatrixXd Model::initWeights() const
{
	int size = getSize();
	MatrixXd newWeights = MatrixXd::Random(size, size) * sqrt(1.0 / size);
	correctWeights(newWeights);
	return newWeights;
}

VectorXd Model::initBias() const
{
	if (isBiasUsed)
	{
		int size = getSize();
		VectorXd newBias = VectorXd::Constant(size, 0);
		correctBias(newBias);
		return newBias;
	}
	else
	{
		return VectorXd();
	}
}

int Model::getSize() const
{
	return vectorCount * windowLength;
}

void Model::correctWeights(MatrixXd& newWeights) const
{
	newWeights = activation(newWeights).cwiseProduct(mask);

	/*auto t1 = chrono::high_resolution_clock::now();*/
	/*int n = getSize();
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (mask(i, j) == 1)
			{
				newWeights(i, j) = tanh(newWeights(i, j));
			}
			else
			{
				newWeights(i, j) = 0;
			}
		}
	}*/
	/*auto t2 = chrono::high_resolution_clock::now();
	cout << "Time : " << ((t2 - t1).count() / 1000000000.0) << endl;*/
}

void Model::correctBias(VectorXd& newBias) const
{
	newBias = activation(newBias);
}

Model::Model(int vectorCount, int windowLength, bool** mask, bool isBiasUsed)
	: vectorCount(vectorCount), windowLength(windowLength), isBiasUsed(isBiasUsed), mask(initMask(mask)), weights(initWeights()), bias(initBias())
{

}

void Model::resetWeights()
{
	weights = initWeights();
}

void Model::resetBias()
{
	bias = initBias();
}
