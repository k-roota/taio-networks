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
	void correctBias(VectorXd &newBias) const;

	template <typename T, int U, int V>
	Matrix<T, U, V> activation(const Matrix<T, U, V>& const x) const
	{
		double tau = 5.0;
		return (tau * x).array().tanh().matrix();
	}
	
public:
	Model(int vectorCount = 1, int windowLength = 3, bool **mask = nullptr, bool isBiasUsed = true);

	void resetWeights();
	void resetBias();
};

