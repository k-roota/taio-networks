#include "Model.h"

#include<chrono>
#include<iostream>

using namespace std;

MatrixXd Model::initMask(shared_ptr<vector<vector<bool>>> mask) const
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
				(*mask)[i][j] ? numMask(i, j) = 1 : numMask(i, j) = 0;
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
	return newWeights.cwiseProduct(mask).cwiseMin(1.0).cwiseMax(-1.0);
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

VectorXd Model::getWindow(const TimeSeries& timeSeries, int id) const
{
	VectorXd X = timeSeries.getWindow(id, windowLength);
	if (activateInput)
		return (*activation)(X);
	else
		return X;
}

Model::Model(const shared_ptr<Activation> activation, const shared_ptr<Loss> loss, const shared_ptr<Optimizer> optimizer, const int vectorCount, const int windowLength, const shared_ptr<vector<vector<bool>>> mask, bool isBiasUsed, double maxLoss, int maxIter, bool activateInput, bool revertOnIncrease, int timeOffset)
	: vectorCount(vectorCount),
	windowLength(windowLength),
	isBiasUsed(isBiasUsed),
	mask(initMask(mask)),
	weights(initWeights()),
	bias(initBias()),
	activation(activation),
	loss(loss),
	optimizer(optimizer),
	maxLoss(maxLoss),
	maxIter(maxIter),
	activateInput(activateInput),
	revertOnIncrease(revertOnIncrease),
	timeOffset(timeOffset)
{}

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

void Model::trainStep(const VectorXd& const X, const VectorXd& const Y, int epoch)
{
	VectorXd A = predictNoAct(X);
	VectorXd YApprox = (*activation)(A);
	if (isBiasUsed)
	{
		VectorXd newB;
		weights = correctWeights((*optimizer)(*loss, *activation, X, A, YApprox, Y, weights, bias, newB, epoch));
		bias = newB;
	}
	else
	{
		weights = correctWeights((*optimizer)(*loss, *activation, X, A, YApprox, Y, weights, epoch));
	}

	//cout << (*loss)(YApprox, Y) << endl;
}

void Model::train(const TimeSeries& timeSeries)
{
	double prevLoss = numeric_limits<double>::max();
	MatrixXd prevWeights;
	VectorXd prevBias;

	for (int epoch = 1; epoch <= maxIter; epoch++)
	{
		if (revertOnIncrease)
		{
			prevWeights = weights;
			prevBias = bias;
		}
		
		VectorXd Y = getWindow(timeSeries, 0);
		for (int i = 1; i <= timeSeries.getLastId(windowLength) - timeOffset + 1; i++)
		{
			VectorXd X = Y;
			Y = getWindow(timeSeries, i + timeOffset - 1);
			//cout << endl << "X" << endl << X << endl << endl << "Y" << endl << Y << endl << endl;
			trainStep(X, Y);
		}
		double epochLoss = evaluate(timeSeries);
		cout << "Epoch: " << epoch << ", Mean loss: " << epochLoss << endl;

		if (epochLoss <= maxLoss)
			break;

		if (revertOnIncrease && epochLoss > prevLoss)
		{
			weights = prevWeights;
			bias = prevBias;
			break;
		}
		prevLoss = epochLoss;
	}
}

double Model::evaluate(const TimeSeries& timeSeries) const
{
	double lossSum = 0.0;
	int windowsCount = timeSeries.getLastId(windowLength) - timeOffset + 1;

	VectorXd Y = getWindow(timeSeries, 0);
	for (int i = 1; i <= windowsCount; i++)
	{
		VectorXd X = Y;
		Y = getWindow(timeSeries, i + timeOffset - 1);
		VectorXd YApprox = predict(X);
		double currLoss = (*loss)(YApprox, Y);
		lossSum += currLoss;
	}
	return lossSum / windowsCount;
}

MatrixXd Model::getWeights() const
{
	return weights;
}

void Model::setWeights(const MatrixXd& const matrix)
{
	weights = correctWeights(matrix);
}

int Model::getWindowLength() const
{
	return windowLength;
}

VectorXd Model::getBias() const
{
	return bias;
}

void Model::setBias(const VectorXd& const vector)
{
	bias = vector;
}

bool Model::checkIsBiasUsed() const
{
	return isBiasUsed;
}
