#pragma once
#include<Eigen/Dense>

using namespace Eigen;

class Model
{
private:
	int vectorCount;
	int windowLength;
	MatrixXd mask;
	MatrixXd weights;
	bool isBiasUsed;
	VectorXd bias;

	MatrixXd initMask(bool **mask) const;
	MatrixXd initWeights() const;
	VectorXd initBias() const;
	int getSize() const;
	void correctWeights(MatrixXd &newWeights) const;
	
public:
	Model(int vectorCount = 1, int windowLength = 3, bool **mask = nullptr, bool isBiasUsed = true);

	void resetWeights();
	void resetBias();
};

