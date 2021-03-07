#pragma once

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <float.h>

/////////////////////////////
/////////////////////////////

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1

#include <CL/cl2.hpp>

/////////////////////////////
/////////////////////////////

using std::cout;
using std::cerr;
using std::endl;

#define OCL_CHECK(error, call)                                                 \
  call;                                                                        \
  if (error != CL_SUCCESS) {                                                   \
    printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__,     \
           __LINE__, error);                                                   \
    exit(EXIT_FAILURE);                                                        \
  }

/////////////////////////////
/////////////////////////////

struct ConfigOpenCL {

	ConfigOpenCL(std::string kernel_name, unsigned int num_kernels = 1, unsigned int num_queues = 1) : kernel_name(kernel_name), num_kernels(num_kernels), num_queues(num_queues) {
		kernel = std::vector<cl::Kernel>(num_kernels);
		queue = std::vector<cl::CommandQueue>(num_queues);
	}

	cl::Kernel get_kernel() { return kernel[0]; }
	cl::CommandQueue get_queue() { return queue[0]; }

	std::string kernel_name;
	std::string xclbin;

	cl::Platform platform;   // platform id;
	cl::Device device;    	 // compute device id;
	cl::Context context;     // compute context;
	std::vector<cl::CommandQueue> queue;  // compute command queue;
	cl::Program program;     // compute program;
	std::vector<cl::Kernel> kernel; // compute kernel;
	unsigned int num_kernels = 1;
	unsigned int num_queues = 1;
};

/////////////////////////////
/////////////////////////////

inline const char *getErrorString(cl_int error) {
	switch (error) {
	// run-time and JIT compiler errors
	case 0:
		return "CL_SUCCESS";
	case -1:
		return "CL_DEVICE_NOT_FOUND";
	case -2:
		return "CL_DEVICE_NOT_AVAILABLE";
	case -3:
		return "CL_COMPILER_NOT_AVAILABLE";
	case -4:
		return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case -5:
		return "CL_OUT_OF_RESOURCES";
	case -6:
		return "CL_OUT_OF_HOST_MEMORY";
	case -7:
		return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case -8:
		return "CL_MEM_COPY_OVERLAP";
	case -9:
		return "CL_IMAGE_FORMAT_MISMATCH";
	case -10:
		return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case -11:
		return "CL_BUILD_PROGRAM_FAILURE";
	case -12:
		return "CL_MAP_FAILURE";
	case -13:
		return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case -14:
		return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case -15:
		return "CL_COMPILE_PROGRAM_FAILURE";
	case -16:
		return "CL_LINKER_NOT_AVAILABLE";
	case -17:
		return "CL_LINK_PROGRAM_FAILURE";
	case -18:
		return "CL_DEVICE_PARTITION_FAILED";
	case -19:
		return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

		// compile-time errors
	case -30:
		return "CL_INVALID_VALUE";
	case -31:
		return "CL_INVALID_DEVICE_TYPE";
	case -32:
		return "CL_INVALID_PLATFORM";
	case -33:
		return "CL_INVALID_DEVICE";
	case -34:
		return "CL_INVALID_CONTEXT";
	case -35:
		return "CL_INVALID_QUEUE_PROPERTIES";
	case -36:
		return "CL_INVALID_COMMAND_QUEUE";
	case -37:
		return "CL_INVALID_HOST_PTR";
	case -38:
		return "CL_INVALID_MEM_OBJECT";
	case -39:
		return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case -40:
		return "CL_INVALID_IMAGE_SIZE";
	case -41:
		return "CL_INVALID_SAMPLER";
	case -42:
		return "CL_INVALID_BINARY";
	case -43:
		return "CL_INVALID_BUILD_OPTIONS";
	case -44:
		return "CL_INVALID_PROGRAM";
	case -45:
		return "CL_INVALID_PROGRAM_EXECUTABLE";
	case -46:
		return "CL_INVALID_KERNEL_NAME";
	case -47:
		return "CL_INVALID_KERNEL_DEFINITION";
	case -48:
		return "CL_INVALID_KERNEL";
	case -49:
		return "CL_INVALID_ARG_INDEX";
	case -50:
		return "CL_INVALID_ARG_VALUE";
	case -51:
		return "CL_INVALID_ARG_SIZE";
	case -52:
		return "CL_INVALID_KERNEL_ARGS";
	case -53:
		return "CL_INVALID_WORK_DIMENSION";
	case -54:
		return "CL_INVALID_WORK_GROUP_SIZE";
	case -55:
		return "CL_INVALID_WORK_ITEM_SIZE";
	case -56:
		return "CL_INVALID_GLOBAL_OFFSET";
	case -57:
		return "CL_INVALID_EVENT_WAIT_LIST";
	case -58:
		return "CL_INVALID_EVENT";
	case -59:
		return "CL_INVALID_OPERATION";
	case -60:
		return "CL_INVALID_GL_OBJECT";
	case -61:
		return "CL_INVALID_BUFFER_SIZE";
	case -62:
		return "CL_INVALID_MIP_LEVEL";
	case -63:
		return "CL_INVALID_GLOBAL_WORK_SIZE";
	case -64:
		return "CL_INVALID_PROPERTY";
	case -65:
		return "CL_INVALID_IMAGE_DESCRIPTOR";
	case -66:
		return "CL_INVALID_COMPILER_OPTIONS";
	case -67:
		return "CL_INVALID_LINKER_OPTIONS";
	case -68:
		return "CL_INVALID_DEVICE_PARTITION_COUNT";

		// extension errors
	case -1000:
		return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
	case -1001:
		return "CL_PLATFORM_NOT_FOUND_KHR";
	case -1002:
		return "CL_INVALID_D3D10_DEVICE_KHR";
	case -1003:
		return "CL_INVALID_D3D10_RESOURCE_KHR";
	case -1004:
		return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
	case -1005:
		return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
	default:
		return "Unknown OpenCL error";
	}
}

/////////////////////////////
/////////////////////////////

#define CHECK_ERR(ans) check_error((ans), __FILE__, __LINE__)
inline int check_error(int ret_val, const char *file, int line) {
	if (ret_val != CL_SUCCESS) {
		std::cerr << "ERROR: " << getErrorString(ret_val) << ", file: " << file << ", line: " << line << std::endl;
		std::cerr.flush();
		exit(EXIT_FAILURE);
	}
	return ret_val;
}

#define CHECK_CREATE_BUFFER(ans) check_create_buffer((ans), __FILE__, __LINE__)
template<typename T>
inline T check_create_buffer(T ret_val, const char *file, int line) {
	if (!ret_val) {
		std::cerr << "ERROR: " << ret_val << ", file: " << file << ", line: " << line << std::endl;
		std::cerr.flush();
		exit(EXIT_FAILURE);
	}
	return ret_val;
}

/////////////////////////////
/////////////////////////////

inline int find_xilinx_device(ConfigOpenCL &config, std::vector<std::string> &target_devices, std::vector<std::string> &kernels, bool debug = true) {
	std::vector<cl::Device> devices;
	std::vector<cl::Platform> platforms;
	bool found_device = false;

	// Traversing all platforms to find a Xilinx platform;
	cl::Platform::get(&platforms);
	if (debug) {
		std::cout << "INFO: Found " << platforms.size() << " platforms" << std::endl;
	}

	for (size_t i = 0; (i < platforms.size()) & (found_device == false); i++) {
		cl::Platform platform = platforms[i];
		std::string platform_name = platform.getInfo<CL_PLATFORM_NAME>();
		if (platform_name == "Xilinx") {
			config.platform = platform;

			if (debug) {
				std::cout << "INFO: Selected platform: " << platform_name << std::endl;
			}

			devices.clear();
			platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);

			// Traverse all devices belonging to the Xilinx platform to select the specified device;
			for (auto target_device_name : target_devices) {
				for (size_t j = 0; j < devices.size(); j++) {
					cl::Device device = devices[j];
					std::string device_name = device.getInfo<CL_DEVICE_NAME>();
					if (device_name == target_device_name) {
						found_device = true;
						config.device = device;
						config.xclbin = kernels[j];
						if (debug) {
							std::cout << "INFO: Selected " << device_name << " as the target device" << std::endl;
							std::cout << "INFO: Selected " << config.xclbin << " as the target xclbin" << std::endl;
						}
						return 0;
					}
				}
			}
		}
	}
	if (found_device == false) {
		if (debug) {
			std::cout << "Error: Unable to find suitable target devices" << std::endl;
		}
		return EXIT_FAILURE;
	} else {
		return 0;
	}
}

inline int setup_opencl_and_load_kernel(ConfigOpenCL &config, bool debug = true) {
	// Creating Context and Command Queue for the selected device;
	config.context = cl::Context(config.device);
	for (int i = 0; i < config.num_queues; i++) {
		config.queue[i] = cl::CommandQueue(config.context, config.device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
	}

	// Load binary from disk
	if (debug) {
		cout << "INFO: Loading " << config.xclbin << endl;
		cout.flush();
	}

	std::ifstream bin_file(config.xclbin, std::ifstream::binary);
	bin_file.seekg(0, bin_file.end);
	uint32_t size = bin_file.tellg();
	bin_file.seekg(0, bin_file.beg);
	char *buffer = new char[size];
	bin_file.read(buffer, size);
	bin_file.close();

	cl::Program::Binaries bins{ { buffer, size } };
	config.program = cl::Program(config.context, std::vector<cl::Device>{ config.device }, bins);

	// A kernel is an OpenCL function that is executed on the FPGA;
	for (unsigned int i = 0; i < config.num_kernels; i++) {
		std::string cu_id = std::to_string(i + 1);
		std::string krnl_name_full = config.kernel_name + ":{" + config.kernel_name + "_" + cu_id + "}";
		if (debug) printf("Creating a kernel [%s] for CU(%d)\n", krnl_name_full.c_str(), i);
		config.kernel[i] = cl::Kernel(config.program, krnl_name_full.c_str());
	}
	return 0;
}

inline int setup_opencl(ConfigOpenCL &config, std::vector<std::string> &target_devices, std::vector<std::string> &kernels, bool debug = true) {
	int err = find_xilinx_device(config, target_devices, kernels, debug);
	err = setup_opencl_and_load_kernel(config, debug);

	if(debug){
		std::cout << "err: " << err << std::endl;
	}

	return err;
}

inline double get_event_execution_time(cl::Event &e) {
	cl_ulong timer_start;
	cl_ulong timer_end;

	e.getProfilingInfo(CL_PROFILING_COMMAND_START, &timer_start);
	e.getProfilingInfo(CL_PROFILING_COMMAND_END, &timer_end);

	return timer_end - timer_start;
}

inline double get_events_execution_time(size_t n_events, ...){
	va_list cl_events;
	va_start(cl_events, n_events);

	double min = DBL_MAX;
	double max = -1.0;

	for(uint i = 0; i < n_events; ++i){

		cl_event cur_event = va_arg(cl_events, cl_event);
		cl_ulong timer_start;
		cl_ulong timer_end;

		clGetEventProfilingInfo(cur_event, CL_PROFILING_COMMAND_START, sizeof(timer_start), &timer_start, NULL);
		clGetEventProfilingInfo(cur_event, CL_PROFILING_COMMAND_END, sizeof(timer_end), &timer_end, NULL);

		if(timer_start < min) min = timer_start;
		if(timer_end > max) max = timer_end;
	}
	return max - min;
}
