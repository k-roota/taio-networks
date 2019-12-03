#include "TimeSeriesConfig.h"

TimeSeriesConfig::TimeSeriesConfig(string matrixType, int dataPortion, int timeOffset):
	matrixType(matrixType),
	dataPortion(dataPortion),
	timeOffset(timeOffset)
{}

vector<vector<bool>> TimeSeriesConfig::getMatrix(int seriesCount) const
{
	int size = seriesCount * dataPortion;
	
	if (matrixType == "CUSTOM")
	{
		cout << "Enter the path to the matrix file" << endl;
		string path;
		getline(cin, path);

		ifstream fileStream(path);
		if (!fileStream.good())
		{
			throw invalid_argument("Invalid path to the matrix file");
		}
		vector<vector<bool>> matrix = MatrixReader::read<bool>(fileStream);
		fileStream.close();

		if (matrix.size() != size || matrix[0].size() != size)
		{
			throw invalid_argument("The matrix in the specified file has wrong size");
		}

		return matrix;
	}
	else
	{
		vector<vector<bool>> matrix;

		for (int i = 0; i < size; i++)
		{
			vector<bool> matrixRow;
			int iTime = i / seriesCount;
			int iSeries = i % seriesCount;

			for (int j = 0; j < size; j++)
			{
				int jTime = j / seriesCount;
				int jSeries = j % seriesCount;
				
				bool value;
				if (matrixType == "FULL")
				{
					value = true;
				}
				else if (matrixType == "CHRONOLOGIC")
				{
					value = ((iTime >= jTime) && (iSeries >= jSeries));
				}
				else if (matrixType == "INDEPENDENT")
				{
					value = ((iTime == jTime) && (iSeries >= jSeries));
				}
				else
				{
					throw invalid_argument("Unknown matrix type");
				}

				matrixRow.push_back(value);
			}

			matrix.push_back(matrixRow);
		}

		return matrix;
	}
}
