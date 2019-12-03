#pragma once

#include <vector>
#include <iostream>
#include <string>
#include <sstream>

using namespace std;

class MatrixReader
{
public:
	template<typename T>
	static vector<vector<T>> read(istream& stream);
};

template<typename T>
inline vector<vector<T>> MatrixReader::read(istream& stream)
{
	vector<vector<T>> matrix;

	while (!stream.eof())
	{
		vector<T> matrixRow;
		string line;
		getline(stream, line);
		istringstream lineStream(line);
		while (!lineStream.eof())
		{
			string value;
			lineStream >> value;
			if (!value.empty())
				matrixRow.push_back(stod(value));
		}
		if(!matrixRow.empty())
			matrix.push_back(matrixRow);
	}
	return matrix;
}
