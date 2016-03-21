#pragma once
#include <stdlib.h>
#include "ReadingData.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

using namespace std;

void ReadingSmallData(std::vector<double>& smallData)
{
	ifstream myLogFile;
	string line;
	myLogFile.open("C:\\Users\\Computing\\Documents\\GitHub\\OpenCL\\OpenCL Tutorials - Tutorial 3 (4)\\OpenCL Tutorials\\temp_lincolnshire_short.txt");

	cout << "Begins Reading Data!" << endl;
	int i = 0;
	for (string line; getline(myLogFile, line);)
	{
		istringstream in(line);
		string type;
		in >> type;
		float a, b, c, d, e;
		in >> a >> b >> c >> d >> e;
		smallData.push_back (e);
		i++;
	}
	cout << "Values Read in: " << i << endl;
	myLogFile.close();
}
