# OpenCL
Parallel Computing

# Profiling
### This will show the excution time of the kernel
- To enable profiling >> `cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);`
- Then you have to create an event: `cl::Event kernel_event;`
- and add it to 
`queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &kernel_event);`
- Then you output the Execution time at the end `std::cout << "Kernel execution time [ns]:" << kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;`

### This will get the full information from the event
`std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << endl;`
