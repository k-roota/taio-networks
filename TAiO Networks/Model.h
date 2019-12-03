#pragma once
#include <Eigen/Dense>
#include <vector>
#include <limits>

#include "Activation.h"
#include "Loss.h"
#include "Optimizer.h"
#include "TimeSeries.h"

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
	const shared_ptr<Activation> activation;
	const shared_ptr<Loss> loss;
	const shared_ptr<Optimizer> optimizer;
	double maxLoss;
	int maxIter;
	bool activateInput;
	bool revertOnIncrease;
	int timeOffset;
	

	MatrixXd initMask(shared_ptr<vector<vector<bool>>> mask) const;
	MatrixXd initWeights() const;
	VectorXd initBias() const;
	int getSize() const;
	MatrixXd correctWeights(const MatrixXd & const newWeights) const;
	VectorXd predictNoAct(const VectorXd& const X) const;
	VectorXd getWindow(const TimeSeries& timeSeries, int id) const;
	
public:
	Model(const shared_ptr<Activation> activation, const shared_ptr<Loss> loss, const shared_ptr<Optimizer> optimizer, const int vectorCount = 1, const int windowLength = 3, const shared_ptr<vector<vector<bool>>> mask = nullptr, bool isBiasUsed = true, double maxLoss = 0.01, int maxIter = 1000, bool activateInput = false, bool revertOnIncrease = false, int timeOffset = 1);

	void resetWeights();
	void resetBias();
	VectorXd predict(const VectorXd& const X) const;
	void trainStep(const VectorXd& const X, const VectorXd& const Y, int epoch = 1);
	void train(const TimeSeries& timeSeries);
	double evaluate(const TimeSeries& timeSeries) const;
	MatrixXd getWeights() const;
	void setWeights(const MatrixXd& const matrix);
	int getWindowLength() const;
	VectorXd getBias() const;
	void setBias(const VectorXd& const vector);
	bool checkIsBiasUsed() const;
};

