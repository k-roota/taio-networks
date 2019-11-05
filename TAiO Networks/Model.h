#pragma once
#include<Eigen/Dense>

#include "Activation.h"
#include "Loss.h"
#include "Optimizer.h"

using namespace Eigen;
using namespace std;

class Model
{
private:
	int vectorCount;
	int windowLength;
	MatrixXd mask;
	MatrixXd weights;
	bool isBiasUsed;
	VectorXd bias;
	Activation* activation;
	Loss* loss;
	Optimizer* optimizer;

	MatrixXd initMask(bool **mask) const;
	MatrixXd initWeights() const;
	VectorXd initBias() const;
	int getSize() const;
	MatrixXd correctWeights(const MatrixXd & const newWeights) const;
	VectorXd predictNoAct(const VectorXd& const X) const;
	
public:
	Model(Activation* const activation, Loss* const loss, Optimizer* const optimizer, int vectorCount = 1, int windowLength = 3, bool **mask = nullptr, bool isBiasUsed = true);

	void resetWeights();
	void resetBias();
	VectorXd predict(const VectorXd& const X) const;
	void trainStep(const VectorXd& const X, const VectorXd& const Y);
	MatrixXd getWeights() const;
};

