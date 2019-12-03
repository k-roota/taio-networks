#pragma once

#include <string>

#include "Loss.h"
#include "Mse.h"

using namespace std;

class NetworkConfig
{
private:
	Loss* createLoss(string lossType) const;

public:
	NetworkConfig(string lossType, double maxLoss, int maxIter, bool activateInput, bool bias, bool revertOnIncrease);

	shared_ptr<Loss> loss;
	double maxLoss;
	int maxIter;
	bool activateInput;
	bool bias;
	bool revertOnIncrease;
};

