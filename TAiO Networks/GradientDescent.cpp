#include "GradientDescent.h"

double GradientDescent::getLearningRate() const
{
	return 0.1;/////////////// 0.1
}

MatrixXd GradientDescent::operator()(const Loss& const loss, const Activation& const activation, const VectorXd& const X, const VectorXd& const A, const VectorXd& const YApprox, const VectorXd& const Y, const MatrixXd& const W) const
{
	return W - getLearningRate() * loss.derivativeW(activation, X, A, YApprox, Y);
}

MatrixXd GradientDescent::operator()(const Loss& const loss, const Activation& const activation, const VectorXd& const X, const VectorXd& const A, const VectorXd& const YApprox, const VectorXd& const Y, const MatrixXd& const W, const VectorXd& const B, VectorXd& outB) const
{
	double lr = getLearningRate();
	VectorXd lossDB = loss.derivativeB(activation, A, YApprox, Y);
	outB = B - lr * lossDB;
	return W - lr * loss.derivativeW(activation, X, A, YApprox, Y, &lossDB);
}

