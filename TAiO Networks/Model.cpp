#include "Model.h"

#include<chrono>
#include<iostream>

using namespace std;

MatrixXd Model::initMask(bool** mask) const
{
	int size = getSize();
	if (mask == nullptr)
	{
		return MatrixXd::Ones(size, size);
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
	return correctWeights(newWeights);
}

VectorXd Model::initBias() const
{
	if (isBiasUsed)
	{
		int size = getSize();
		VectorXd newBias = VectorXd::Zero(size);
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

MatrixXd Model::correctWeights(const MatrixXd & const newWeights) const
{
	return newWeights.cwiseProduct(mask);
}

VectorXd Model::predictNoAct(const VectorXd& const X) const
{
	if (isBiasUsed)
	{
		return weights * X + bias;
	}
	else
	{
		return weights * X;
	}
}

Model::Model(Activation* const activation, Loss* const loss, Optimizer* const optimizer, int vectorCount, int windowLength, bool** mask, bool isBiasUsed)
	: vectorCount(vectorCount), windowLength(windowLength), isBiasUsed(isBiasUsed), mask(initMask(mask)), weights(initWeights()), bias(initBias()),
		activation(activation), loss(loss), optimizer(optimizer)
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

VectorXd Model::predict(const VectorXd& const X) const
{
	return (*activation)(predictNoAct(X));
}

void Model::trainStep(const VectorXd& const X, const VectorXd& const Y)
{
	VectorXd A = predictNoAct(X);
	VectorXd YApprox = (*activation)(A);
	if (isBiasUsed)
	{
		VectorXd newB;
		weights = correctWeights((*optimizer)(*loss, *activation, X, A, YApprox, Y, weights, bias, newB));
		bias = newB;
	}
	else
	{
		weights = correctWeights((*optimizer)(*loss, *activation, X, A, YApprox, Y, weights));
	}

	cout << (*loss)(YApprox, Y) << endl;
}

MatrixXd Model::getWeights() const
{
	return weights;
}
