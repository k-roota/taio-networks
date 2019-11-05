#include "Mse.h"

//double Mse::operator()(const VectorXd& const YApprox, const VectorXd& const Y) const
//{
//	VectorXd test = Y.cwiseInverse();//////////////////
//	auto difference = (Y - YApprox).cwiseProduct(test);
//	return difference.cwiseProduct(difference).mean();
//}

double Mse::operator()(const VectorXd& const YApprox, const VectorXd& const Y) const
{
	auto difference = (Y - YApprox);
	return difference.cwiseProduct(difference).mean();
}

//VectorXd Mse::derivativeY(const VectorXd& const YApprox, const VectorXd& const Y) const
//{
//	int n = Y.size();
//	VectorXd test = Y.cwiseInverse();/////////////////////
//	test = test.cwiseProduct(test);
//	return (2.0 / n * (YApprox - Y)).cwiseProduct(test);
//}

VectorXd Mse::derivativeY(const VectorXd& const YApprox, const VectorXd& const Y) const
{
	int n = Y.size();
	return (2.0 / n * (YApprox - Y));
}

VectorXd Mse::derivativeB(const Activation& const activation, const VectorXd& const A, const VectorXd& const YApprox, const VectorXd& const Y) const
{
	auto activationDA = activation.derivativeA(A, &YApprox);
	return derivativeY(YApprox, Y).cwiseProduct(activationDA);
}

MatrixXd Mse::derivativeW(const Activation& const activation, const VectorXd& const X, const VectorXd& const A, const VectorXd& const YApprox, const VectorXd& const Y, const VectorXd* const lossDB) const
{
	if (lossDB == nullptr)
	{
		return derivativeB(activation, A, YApprox, Y) * X.transpose();
	}
	else
	{
		return *lossDB * X.transpose();
	}
}
