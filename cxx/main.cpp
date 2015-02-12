#include <CL/cl.h>

#include <cstdio>
#include <cassert>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>

// small helpers
inline void except(bool condition, const std::string &error_message = "")
{
	if (!condition)
		throw std::runtime_error(error_message);
}

size_t grid_dim(size_t block_dim, size_t array_dim)
{
	size_t ret = array_dim / block_dim;
	if (array_dim % block_dim)
		return ret + 1;
	else
		return ret;
}

// assume all matrixes as plain arrays with column-major storage

// for simplicity suppose that we know N and M before reading file
// of course it can be changed if needed
std::vector<char> read_file(const std::string &file_name, int rows, int cols)
{
	std::vector<char> result(rows * cols);
	FILE *fp = fopen(file_name.c_str(), "rb");
	except(fp, "can't open input file '" + file_name + "'");
	try
	{
		for (int r = 0; r < rows; ++r)
		{
			for (int c = 0; c < cols; ++c)
			{
				except(fscanf(fp, "%c ", &result[r * cols + c]) == 1,
					   "incorrect matrix in '" + file_name + "'");
			}
		}
	}
	catch (const std::runtime_error &)
	{
		if (fp)
			fclose(fp);
		throw;
	}
	return result;
}

class MatFinder
{
public:
	MatFinder()
		:
		source_str(0),
		source_size(0),
		command_queue(0),
		search_k(0),
		search_row_k(0),
		memset_k(0),
		program(0),
		input_d(0),
		waldo_d(0),
		output_d(0),
		context(0)
	{
		set_defaults();
	}

	~MatFinder()
	{
		if (command_queue)
		{
			clFlush(command_queue);
			clFinish(command_queue);
		}
		clean_buffers();
		if (search_k)
			clReleaseKernel(search_k);
		search_k = 0;
		if (search_row_k)
			clReleaseKernel(search_row_k);
		search_row_k = 0;
		if (memset_k)
			clReleaseKernel(memset_k);
		memset_k = 0;
		if (program)
			clReleaseProgram(program);
		program = 0;
		if (command_queue)
			clReleaseCommandQueue(command_queue);
		command_queue = 0;
		if (context)
			clReleaseContext(context);
		context = 0;
		delete source_str;
		source_str = 0;
	}

	// set default options
	void set_defaults()
	{
		synchronous_tasks = true;
		verbose = true;
		kernel_file_name = "cl/mat.cl";
		k_rect_name = "search_simple";
		k_row_name = "search_by_row";
		k_memset_name = "memset_int";
		simple_branch = true;

		input_mat_file_name = "data/input.mat";
		search_mat_file_name = "data/search.mat";
		input_mat_rows = 4;
		input_mat_cols = 7;
		search_mat_rows = 2;
		search_mat_cols = 2;
		output_rows = input_mat_rows;
		output_cols = input_mat_cols;
	}

	void setup()
	{
		assert(!source_str);  // don't call setup() twice

		source_str = new char[0x10000];
		FILE *fp = fopen(kernel_file_name.c_str(), "rb");
		if (!fp)
			fp = fopen(("../" + kernel_file_name).c_str(), "rb");
		except(fp, "kernel file '" + kernel_file_name + "' doesn't exist");
		source_size = fread(source_str, 1, 0x10000, fp);
		fclose(fp);

		cl_uint num_platforms = 0;
		cl_platform_id platform_id = 0;
		cl_int ret = clGetPlatformIDs(1, &platform_id, &num_platforms);
		except(ret == CL_SUCCESS, "can't get platform info");
		except(num_platforms, "no OpenCL platforms found");
		cl_device_id device_id = 0;
		cl_uint num_devices = 0;
		ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL,
			0, 0, &num_devices);
		except(ret == CL_SUCCESS, "can't devices list");
		except(num_devices, "no OpenCL devices found");
		cl_uint num_default_device = 0;
		ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT,
			1, &device_id, &num_default_device);
		except(ret == CL_SUCCESS, "can't get device info");
		if (verbose)
		{
			char info_str[256];
			printf("Detected %d platforms\n", num_platforms);
			ret = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME,
				sizeof(info_str), info_str, NULL);
			printf("OpenCL platform: %s, ", info_str);
			ret = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION,
				sizeof(info_str), info_str, NULL);
			printf("version %s\n", info_str);
			printf("Detected %d devices\n", num_devices);
			for (int i = 0; i < num_devices; ++i)
			{
				// get info for all devices - not necessary now
			}

			size_t dev_param_size = 0;
			ret = clGetDeviceInfo(device_id, CL_DEVICE_NAME,
				sizeof(info_str), info_str, &dev_param_size);
			printf("Default device: %s\n", info_str);
			size_t local_size = 0;
			ret = clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE,
				sizeof(local_size), &local_size, &dev_param_size);
			printf("Block mem size: %dKb\n", (int)local_size / 1024);
		}

		context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
		except(ret == CL_SUCCESS, "can't create device context");

		if (synchronous_tasks)
			command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
		else
			command_queue = clCreateCommandQueue(context, device_id,
				CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &ret);
		except(ret == CL_SUCCESS, "can't create command queue");

		program = clCreateProgramWithSource(context, 1,
			(const char **)&source_str, (const size_t *)&source_size, &ret);
		except(ret == CL_SUCCESS, "can't read program");

		ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
		except(ret == CL_SUCCESS, "can't build program");

		memset_k = clCreateKernel(program, "memset_int", &ret);
		except(ret == CL_SUCCESS, "can't create memset kernel");
		search_k = clCreateKernel(program, k_rect_name.c_str(), &ret);
		except(ret == CL_SUCCESS, "can't create search kernel");
		search_row_k = clCreateKernel(program, k_row_name.c_str(), &ret);
		except(ret == CL_SUCCESS, "can't create search row kernel");
	}

	void alloc_buffers()
	{
		assert(!output_d);  // clean buffers before new allocation!

		cl_int ret = CL_SUCCESS;
		input_d = clCreateBuffer(context,
			CL_MEM_READ_WRITE,
			input_mat_rows * input_mat_cols * sizeof(char), NULL, &ret);
		except(ret == CL_SUCCESS, "can't create input memory object");
		waldo_d = clCreateBuffer(context,
			CL_MEM_READ_WRITE,
			search_mat_rows * search_mat_cols * sizeof(char), NULL, &ret);
		except(ret == CL_SUCCESS, "can't create search memory object");
		output_d = clCreateBuffer(context,
			CL_MEM_READ_WRITE,
			output_rows * output_cols * sizeof(int), NULL, &ret);
		except(ret == CL_SUCCESS, "can't create output memory object");
	}

	void clean_buffers()
	{
		if (input_d)
			clReleaseMemObject(input_d);
		input_d = 0;
		if (waldo_d)
			clReleaseMemObject(waldo_d);
		waldo_d = 0;
		if (output_d)
			clReleaseMemObject(output_d);
		output_d = 0;
	}

	void read_data()
	{
		// must allocate buffers before reading
		assert(input_d && waldo_d && output_d);

		std::vector<char> input_mat =
			read_file(input_mat_file_name, input_mat_rows, input_mat_cols);
		std::vector<char> search_mat =
			read_file(search_mat_file_name, search_mat_rows, search_mat_cols);

		cl_int ret = clEnqueueWriteBuffer(command_queue, input_d, CL_TRUE, 0,
			input_mat_rows * input_mat_cols * sizeof(char),
			&input_mat[0], 0, NULL, NULL);
		except(ret == CL_SUCCESS, "can't copy input to device");
		ret = clEnqueueWriteBuffer(command_queue, waldo_d, CL_TRUE, 0,
			search_mat_rows * search_mat_cols * sizeof(char),
			&search_mat[0], 0, NULL, NULL);
		except(ret == CL_SUCCESS, "can't copy search tile to device");
	}

	void set_data(const std::vector<char> &input_mat,
				  const std::vector<char> &search_mat)
	{
		// must allocate buffers before reading
		assert(input_d && waldo_d && output_d);

		cl_int ret = clEnqueueWriteBuffer(command_queue, input_d, CL_TRUE, 0,
			input_mat_rows * input_mat_cols * sizeof(char),
			&input_mat[0], 0, NULL, NULL);
		except(ret == CL_SUCCESS, "can't copy input to device");
		ret = clEnqueueWriteBuffer(command_queue, waldo_d, CL_TRUE, 0,
			search_mat_rows * search_mat_cols * sizeof(char),
			&search_mat[0], 0, NULL, NULL);
		except(ret == CL_SUCCESS, "can't copy search tile to device");
	}

	void perform_rect_search()
	{
		simple_branch = true;

		// must allocate all internals here
		assert(search_row_k);
		// must allocate buffers here
		assert(input_d && waldo_d && output_d);

		size_t block_size[3] = { 16, 16, 0 };
		size_t grid_size[3] = {
			block_size[0] * grid_dim(block_size[0], input_mat_cols - search_mat_cols + 1),
			block_size[1] * grid_dim(block_size[1], input_mat_rows - search_mat_rows + 1),
			0
		};

		cl_int ret = clSetKernelArg(search_k, 0, sizeof(cl_mem), (void *)&output_d);
		except(ret == CL_SUCCESS, "can't set kernel arguments (0)");
		ret = clSetKernelArg(search_k, 1, sizeof(cl_mem), (void *)&input_d);
		except(ret == CL_SUCCESS, "can't set kernel arguments (1)");
		ret = clSetKernelArg(search_k, 2, sizeof(int), (void *)&input_mat_rows);
		except(ret == CL_SUCCESS, "can't set kernel arguments (2)");
		ret = clSetKernelArg(search_k, 3, sizeof(int), (void *)&input_mat_cols);
		except(ret == CL_SUCCESS, "can't set kernel arguments (3)");
		ret = clSetKernelArg(search_k, 4, sizeof(cl_mem), (void *)&waldo_d);
		except(ret == CL_SUCCESS, "can't set kernel arguments (4)");
		ret = clSetKernelArg(search_k, 5, sizeof(int), (void *)&search_mat_rows);
		except(ret == CL_SUCCESS, "can't set kernel arguments (5)");
		ret = clSetKernelArg(search_k, 6, sizeof(int), (void *)&search_mat_cols);
		except(ret == CL_SUCCESS, "can't set kernel arguments (6)");

		ret = clEnqueueNDRangeKernel(command_queue, search_k, 2, NULL,
			grid_size, block_size, 0, NULL, NULL);
		except(ret == CL_SUCCESS, "can't launch rect kernel");
	}

	void perform_row_search()
	{
		simple_branch = false;

		// must allocate all internals here
		assert(search_row_k);
		// must allocate buffers here
		assert(input_d && waldo_d && output_d);

		// for OpenCL 1.0 there's simple fill kernel
		int zero_val = 0;
		int dim = output_rows * output_cols;
		size_t block_size_1d = 256;
		size_t grid_size_1d = block_size_1d * grid_dim(block_size_1d, dim);
		cl_int ret = clSetKernelArg(memset_k, 0, sizeof(cl_mem), (void *)&output_d);
		except(ret == CL_SUCCESS, "can't set kernel arguments (0)");
		ret = clSetKernelArg(memset_k, 1, sizeof(int), (void *)&zero_val);
		except(ret == CL_SUCCESS, "can't set kernel arguments (1)");
		ret = clSetKernelArg(memset_k, 2, sizeof(int), (void *)&dim);
		except(ret == CL_SUCCESS, "can't set kernel arguments (2)");
		ret = clEnqueueNDRangeKernel(command_queue, memset_k, 1, NULL,
			&grid_size_1d, &block_size_1d, 0, NULL, NULL);

		// OpenCL 1.2
//			ret = clEnqueueFillBuffer(command_queue,
//				output_d, &val, sizeof(int), 0,
//				output_rows * output_cols * sizeof(int),
//				0, NULL, NULL);
		except(ret == CL_SUCCESS, "can't fill memory by zeros");

		size_t block_size[3] = { 16, 16, 0 };
		size_t grid_size[3] = {
			block_size[0] * grid_dim(block_size[0], input_mat_cols - search_mat_cols + 1),
			block_size[1] * grid_dim(block_size[1], input_mat_rows - search_mat_rows + 1),
			0
		};

		for (int r = 0; r < search_mat_rows; ++r)
		{
			ret = clSetKernelArg(search_row_k, 0, sizeof(cl_mem), (void *)&output_d);
			except(ret == CL_SUCCESS, "can't set kernel arguments (0)");
			ret = clSetKernelArg(search_row_k, 1, sizeof(cl_mem), (void *)&input_d);
			except(ret == CL_SUCCESS, "can't set kernel arguments (1)");
			ret = clSetKernelArg(search_row_k, 2, sizeof(int), (void *)&input_mat_rows);
			except(ret == CL_SUCCESS, "can't set kernel arguments (2)");
			ret = clSetKernelArg(search_row_k, 3, sizeof(int), (void *)&input_mat_cols);
			except(ret == CL_SUCCESS, "can't set kernel arguments (3)");
			ret = clSetKernelArg(search_row_k, 4, sizeof(cl_mem), (void *)&waldo_d);
			except(ret == CL_SUCCESS, "can't set kernel arguments (4)");
			ret = clSetKernelArg(search_row_k, 5, sizeof(int), (void *)&search_mat_rows);
			except(ret == CL_SUCCESS, "can't set kernel arguments (5)");
			ret = clSetKernelArg(search_row_k, 6, sizeof(int), (void *)&search_mat_cols);
			except(ret == CL_SUCCESS, "can't set kernel arguments (6)");
			ret = clSetKernelArg(search_row_k, 7, sizeof(int), (void *)&r);
			except(ret == CL_SUCCESS, "can't set kernel arguments (7)");

			ret = clEnqueueNDRangeKernel(command_queue, search_row_k, 2, NULL,
				grid_size, block_size, 0, NULL, NULL);
			except(ret == CL_SUCCESS, "can't launch row kernel");
		}
	}

	void perform_best()
	{
		if (search_mat_cols >= 32 && search_mat_rows >= 32)
		{
			simple_branch = false;
			perform_row_search();
		}
		else
		{
			simple_branch = true;
			perform_rect_search();
		}
	}

	std::vector<std::pair<int, int> > get_output()
	{
		std::vector<int> output(input_mat_rows * input_mat_cols);
		cl_int ret = clEnqueueReadBuffer(command_queue, output_d, CL_TRUE, 0,
			input_mat_rows * input_mat_cols * sizeof(int),
			&output[0], 0, NULL, NULL);
		except(ret == CL_SUCCESS, "can't read output");

		int comp_val = 1;
		if (!simple_branch)
			comp_val = search_mat_rows;
		std::vector<std::pair<int, int> > result;
		for (int r = 0; r <= input_mat_rows - search_mat_rows; ++r)
		{
			for (int c = 0; c <= input_mat_cols - search_mat_cols; ++c)
			{
				//printf("%d\n", output[r * output_cols + c]);
				if (output[r * output_cols + c] == comp_val)
					result.push_back(std::pair<int, int>(r, c));
			}
		}
		return result;
	}

	// just some options, compile-time for simplicity
	// don't try asynchronous tasks:
	// for this sample kernel concurrent launch
	// is not necessary, and I didn't implement event-based control flow
	bool synchronous_tasks;
	bool verbose;
	std::string kernel_file_name;
	std::string k_rect_name;
	std::string k_row_name;
	std::string k_memset_name;
	std::string input_mat_file_name;
	std::string search_mat_file_name;

	// use simple kernel in case of low search matrix dimensions:
	// it will be faster
	// in case of high 'Waldo' dimensions row search is much better
	bool simple_branch;

	// input data params, call buffers reallocation after changing
	int input_mat_rows;
	int input_mat_cols;
	int search_mat_rows;
	int search_mat_cols;
	int output_rows;
	int output_cols;

private:
	// internal params
	char *source_str;
	size_t source_size;
	cl_command_queue command_queue;
	cl_kernel search_k;
	cl_kernel search_row_k;
	cl_kernel memset_k;
	cl_program program;
	cl_mem input_d;
	cl_mem waldo_d;
	cl_mem output_d;
	cl_context context;
};

// simple tests (don't use GTEST - project is only sample task)
void test_correctness_rect(MatFinder &finder)
{
	printf("correctness (rect): ");

	int rows = 4;
	int cols = 7;
	int srows = 2;
	int scols = 2;
	finder.clean_buffers();
	finder.input_mat_rows = rows;
	finder.input_mat_cols = cols;
	finder.search_mat_rows = srows;
	finder.search_mat_cols = scols;
	finder.output_rows = rows;
	finder.output_cols = cols;
	finder.alloc_buffers();
	std::vector<char> input(rows * cols);
	memcpy(&input[0],
			"12YDABC"
			"42AFYD0"
			"HYDSIT0"
			"UITDSMA",
			rows * cols);
	std::vector<char> waldo(srows * scols);
	memcpy(&waldo[0],
			"YD"
			"IT",
			srows * scols);
	finder.set_data(input, waldo);
	finder.perform_rect_search();

	std::map<int, int> gt;
	gt[1] = 4;
	gt[2] = 1;
	std::vector<std::pair<int, int> > results = finder.get_output();

	bool ok = true;
	for (std::vector<std::pair<int, int> >::const_iterator it = results.begin();
		 it != results.end(); ++it)
	{
		std::map<int, int>::iterator git = gt.find(it->first);
		if (git == gt.end() || git->second != it->second)
		{
			ok = false;
			break;
		}
		gt.erase(git);
	}
	ok &= gt.empty();  // all results found
	if (ok)
		printf("PASSED\n");
	else
		printf("FAILED\n");
}

void test_correctness_row(MatFinder &finder)
{
	printf("correctness (row): ");

	int rows = 4;
	int cols = 7;
	int srows = 2;
	int scols = 2;
	finder.clean_buffers();
	finder.input_mat_rows = rows;
	finder.input_mat_cols = cols;
	finder.search_mat_rows = srows;
	finder.search_mat_cols = scols;
	finder.output_rows = rows;
	finder.output_cols = cols;
	finder.alloc_buffers();
	std::vector<char> input(rows * cols);
	memcpy(&input[0],
			"12YDABC"
			"42AFYD0"
			"HYDSIT0"
			"UITDSMA",
			rows * cols);
	std::vector<char> waldo(srows * scols);
	memcpy(&waldo[0],
			"YD"
			"IT",
			srows * scols);
	finder.set_data(input, waldo);
	finder.perform_row_search();

	std::map<int, int> gt;
	gt[1] = 4;
	gt[2] = 1;
	std::vector<std::pair<int, int> > results = finder.get_output();

	bool ok = true;
	for (std::vector<std::pair<int, int> >::const_iterator it = results.begin();
		 it != results.end(); ++it)
	{
		std::map<int, int>::iterator git = gt.find(it->first);
		if (git == gt.end() || git->second != it->second)
		{
			ok = false;
			break;
		}
		gt.erase(git);
	}
	ok &= gt.empty();  // all results found
	if (ok)
		printf("PASSED\n");
	else
		printf("FAILED\n");
}

void test_correctness_large_matrix(MatFinder &finder)
{
	printf("correctness (large): ");
	int rows = 1000;
	int cols = 1000;
	int srows = 500;
	int scols = 500;
	finder.clean_buffers();
	finder.input_mat_rows = rows;
	finder.input_mat_cols = cols;
	finder.search_mat_rows = srows;
	finder.search_mat_cols = scols;
	finder.output_rows = rows;
	finder.output_cols = cols;
	finder.alloc_buffers();
	std::vector<char> input(rows * cols);
	memset(&input[0], '1', rows * cols);
	std::vector<char> waldo(srows * scols);
	memset(&waldo[0], '2', srows * scols);
	for (int r = 20; r < 20 + srows; ++r)
		for (int c = 51; c < 51 + scols; ++c)
			input[r * cols + c] = '2';
	finder.set_data(input, waldo);
	finder.perform_rect_search();

	std::map<int, int> gt;
	gt[20] = 51;
	std::vector<std::pair<int, int> > results = finder.get_output();

	bool ok = true;
	for (std::vector<std::pair<int, int> >::const_iterator it = results.begin();
		 it != results.end(); ++it)
	{
		std::map<int, int>::iterator git = gt.find(it->first);
		if (git == gt.end() || git->second != it->second)
		{
			ok = false;
			break;
		}
		gt.erase(git);
	}
	ok &= gt.empty();  // all results found
	if (ok)
		printf("PASSED\n");
	else
		printf("FAILED\n");
}

// performace test now only row vs rect (just for demonstration)
// simplest timer, really it's extremely wrong and inaccurate for this task
// it's just method demostration, in real app use profilers
void test_performance_large_matrix(MatFinder &finder)
{
	printf("performance: ");
	int rows = 1000;
	int cols = 1000;
	int srows = 500;
	int scols = 500;
	finder.clean_buffers();
	finder.input_mat_rows = rows;
	finder.input_mat_cols = cols;
	finder.search_mat_rows = srows;
	finder.search_mat_cols = scols;
	finder.output_rows = rows;
	finder.output_cols = cols;
	finder.alloc_buffers();
	std::vector<char> input(rows * cols);
	memset(&input[0], '1', rows * cols);
	std::vector<char> waldo(srows * scols);
	memset(&waldo[0], '2', srows * scols);
	for (int r = 20; r < 20 + srows; ++r)
		for (int c = 51; c < 51 + scols; ++c)
			input[r * cols + c] = '2';
	finder.set_data(input, waldo);

	// simplest timer, really it's extremely wrong and inaccurate for this task
	time_t time0 = time(0);
	finder.perform_row_search();
	time0 = time(0) - time0;
	time_t time1 = time(0);
	finder.perform_rect_search();
	time1 = time(0) - time1;
	printf("%lf vs %lf\n", double(time0) / 1000., double(time1) / 1000.);
}

int main(int argc, char *argv[])
{
	MatFinder finder;
	try
	{
		// conventional result
		finder.setup();
		finder.alloc_buffers();
		finder.read_data();
		finder.perform_rect_search();
		std::vector<std::pair<int, int> > results = finder.get_output();
		printf("RESULT:\n");
		if (results.empty())
			printf("not found\n");
		else
		{
			for (std::vector<std::pair<int, int> >::const_iterator it = results.begin();
				 it != results.end(); ++it)
			{
				printf("%d, %d\n", it->first, it->second);
			}
		}

		printf("running tests...\n");
		test_correctness_rect(finder);
		test_correctness_row(finder);
		test_correctness_large_matrix(finder);
		test_performance_large_matrix(finder);
		printf("OK, shutting down\n");
	}
	catch (const std::runtime_error &e)
	{
		printf("ERROR: %s\n", e.what());
	}

	return 0;
}
