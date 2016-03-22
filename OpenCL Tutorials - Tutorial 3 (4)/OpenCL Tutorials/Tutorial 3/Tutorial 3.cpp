#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include "ReadingData.h"

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels3.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		typedef double mytype;
		typedef int mytype2;

		// ======
		// Input
		// ======
		std::vector<mytype> A = {8.2, 9.1, 7, -5, 1, 4, -2, 2, 3, 6};
		std::vector<mytype> sDT;
		std::vector<mytype> lDT;
		ReadingSmallData(sDT);
		std::vector<mytype2> sDTMean;
		std::vector<mytype> outputsDT;
		cout << "Lets Go!" << endl;

		for (int i = 0; i < 18732; i++)
		{
			sDTMean.push_back (sDT[i] * 10);
		}

		//cout << sDTMean << endl;

		// ================
		// Assignment Data
		// ================
		//size_t newLocal_size = 223;
		//size_t newPadding_size = sDT.size() % newLocal_size;

		//if (newPadding_size)
		//{
		//	std::vector<int> smallDataType_ext(newLocal_size - newPadding_size, 0);
		//	sDT.insert(sDT.end(), smallDataType_ext.begin(), smallDataType_ext.end());
		//}

		//size_t newInput_elements = sDT.size();
		//cout << "How many in Array: " << sDT.size() << endl;
		//size_t newInput_size = sDT.size()*sizeof(mytype2);
		//size_t newNr_groups = newInput_size / newLocal_size;

		//std::vector<mytype2> newOutput(newInput_elements);
		//size_t newOutput_size = newOutput.size()*sizeof(mytype2);

		// ---------
		// For Mean
		// ---------

		size_t newLocal_size = 223;
		size_t newPadding_size = sDTMean.size() % newLocal_size;

		if (newPadding_size)
		{
			std::vector<int> smallDataType_ext(newLocal_size - newPadding_size, 0);
			sDTMean.insert(sDTMean.end(), smallDataType_ext.begin(), smallDataType_ext.end());
		}

		size_t newInput_elements = sDTMean.size();
		cout << "How many in Array: " << sDTMean.size() << endl;
		size_t newInput_size = sDTMean.size()*sizeof(mytype2);
		size_t newNr_groups = newInput_size / newLocal_size;

		std::vector<mytype2> newOutput(newInput_elements);
		size_t newOutput_size = newOutput.size()*sizeof(mytype2);

		// =============
		// Regular Data
		// =============
		//size_t local_size = 10;
		//size_t padding_size = A.size() % local_size;

		//if (padding_size) 
		//{
		//	std::vector<int> A_ext(local_size-padding_size, 0);
		//	A.insert(A.end(), A_ext.begin(), A_ext.end());
		//}

		//size_t input_elements = A.size();
		//cout << "How many in Array: " << smallDataType.size() << endl;
		//size_t input_size = A.size()*sizeof(mytype);
		//size_t nr_groups = input_elements / local_size;

		//std::vector<mytype> B(input_elements);
		//size_t output_size = B.size()*sizeof(mytype);

		// ==================
		// Kernel Events New
		// ==================
		cl::Event kernel_event;
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, newInput_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, newOutput_size);
		
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, newInput_size, &sDTMean[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, newOutput_size);

		// ==================
		// Kernel Events Old
		// ==================
		//cl::Event kernel_event;
		//cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		//cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);

		//queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		//queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);

		// ===================
		// Excute the kernals
		// ===================
		//cl::Kernel kernel_1 = cl::Kernel(program, "reduce_max");
		//cl::Kernel kernel_1 = cl::Kernel(program, "reduce_min");
		//cl::Kernel kernel_1 = cl::Kernel(program, "reduce_Aver");
		cl::Kernel kernel_1 = cl::Kernel(program, "reduce_average");

		// =======================
		// More Kernel Events Old
		// =======================
		//kernel_1.setArg(0, buffer_A);
		//kernel_1.setArg(1, buffer_B);
		//kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));

		//queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &kernel_event);
		//queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

		// =======================
		// More Kernel Events New
		// =======================
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(newLocal_size*sizeof(mytype2)));

		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(newInput_elements), cl::NDRange(newLocal_size), NULL, &kernel_event);
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, newOutput_size, &newOutput[0]);

		//cout << newOutput << endl;

		for (int j = 0; j < 18732; j++)
		{
			outputsDT.push_back (newOutput[j] / 10);
		}
		outputsDT[0] = outputsDT[0] / newOutput.size();

		// =======
		// Output
		// =======
		//std::cout << "A = " << sDT << std::endl;
		std::cout << "B = " << outputsDT[0] << std::endl;
		//std::cout << "A = " << A << std::endl;
		//std::cout << "B = " << B << std::endl;

		std::cout << "Kernel execution time [ns]:" << kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	system("pause");
	return 0;
}
