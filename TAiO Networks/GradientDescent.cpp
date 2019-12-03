#include "GradientDescent.h"

double GradientDescent::getLearningRate(int epoch) const
{
	return 0.01 / sqrt(epoch);
}

MatrixXd GradientDescent::operator()(const Loss& const loss, const Activation& const activation, const VectorXd& const X, const VectorXd& const A, const VectorXd& const YApprox, const VectorXd& const Y, const MatrixXd& const W, int epoch) const
{
	return W - getLearningRate(epoch) * loss.derivativeW(activation, X, A, YApprox, Y);
}

MatrixXd GradientDescent::operator()(const Loss& const loss, const Activation& const activation, const VectorXd& const X, const VectorXd& const A, const VectorXd& const YApprox, const VectorXd& const Y, const MatrixXd& const W, const VectorXd& const B, VectorXd& outB, int epoch) const
{
	double lr = getLearningRate(epoch);
	VectorXd lossDB = loss.derivativeB(activation, A, YApprox, Y);
	outB = B - lr * lossDB;
	return W - lr * loss.derivativeW(activation, X, A, YApprox, Y, &lossDB);
}

