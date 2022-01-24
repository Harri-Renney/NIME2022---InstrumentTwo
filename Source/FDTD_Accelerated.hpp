#ifndef FDTD_ACCELERATED_HPP
#define FDTD_ACCELERATED_HPP

#include <utility>
#include <stdint.h>
#include <iostream>
#include <fstream>

//#define CL_HPP_TARGET_OPENCL_VERSION 210
//#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl2.hpp>
//#include <CL/cl.hpp>
#include <CL/cl_gl.h>

//Parsing parameters as json file//
#include "json.hpp"
using nlohmann::json;

#include "FDTD_Grid.hpp"
#include "Buffer.hpp"

#include "Visualizer.hpp"

#include <chrono>
#include <thread>


#define MAX_CONNECTIONS 100

enum DeviceType { INTEGRATED = 32902, DISCRETE = 4098, NVIDIA = 4318 };
enum Implementation { OPENCL, CUDA, VULKAN, DIRECT3D };

#include <string>

class FDTD_Accelerated
{
private:
	Implementation implementation_;
	uint32_t sampleRate_;
	DeviceType deviceType_;

	//CL//
	cl_int errorStatus_ = 0;
	cl_uint num_platforms, num_devices;
	cl::Platform platform_;
	cl::Context context_;
	cl::Device device_;
	cl::CommandQueue commandQueue_;
	cl::Program kernelProgram_;
	std::string kernelSourcePath_;
	cl::Kernel kernelScheme_;
	cl::Kernel kernelConnections_;
	cl::NDRange globalws_;
	cl::NDRange localws_;

	//CL Buffers//
	cl::Buffer idGrid_;
	cl::Buffer modelGrid_;
	cl::Buffer boundaryGridBuffer_;
	cl::Buffer outputBuffer_;
	cl::Buffer excitationBuffer_;
	cl::Buffer localBuffer_;
	cl::Buffer inputPositionBuffer_;
	cl::Buffer outputPositionBuffer_;
	cl::Buffer connectionsBuffer_;
	cl::Buffer coeffBuffer_;

	//Model//
	int listenerPosition_[2];
	int excitationPosition_[2];
	Model* model_;
	int modelWidth_;
	int modelHeight_;
	int gridElements_;
	int gridByteSize_;
	int numConnections_;
	int* connections_;

	//Output and excitations//
	typedef float base_type_;
	unsigned int bufferSize_;
	Buffer<base_type_> output_;
	Buffer<base_type_> excitation_;

	int bufferRotationIndex_ = 1;

	Visualizer* vis;

	float* renderGrid;
	int* idGridInput_;

	const int maxCoeffs_ = 200;
	const int maxOutputs_ = 100;
	int outputCounter_ = 1;
	int* inputGridInput_;
	int* outputGridInput_;
	float* boundaryGridInput_;
	float emptyBuffer_[44100];

	void initOpenCL()
	{
		std::vector <cl::Platform> platforms;
		cl::Platform::get(&platforms);
		for (cl::vector<cl::Platform>::iterator it = platforms.begin(); it != platforms.end(); ++it)
		{
			cl::Platform platform(*it);

			cl_context_properties contextProperties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0 };
			context_ = cl::Context(CL_DEVICE_TYPE_GPU, contextProperties);

			cl::vector<cl::Device> devices = context_.getInfo<CL_CONTEXT_DEVICES>();

			int device_id = 0;
			for (cl::vector<cl::Device>::iterator it2 = devices.begin(); it2 != devices.end(); ++it2)
			{
				cl::Device device(*it2);
				auto d = device.getInfo<CL_DEVICE_VENDOR_ID>();
				if (d == DeviceType::NVIDIA || d == DeviceType::DISCRETE)	//Hard coded to pick AMD or NVIDIA.
				{
					//Create command queue for first device - Profiling enabled//
					commandQueue_ = cl::CommandQueue(context_, device, CL_QUEUE_PROFILING_ENABLE, &errorStatus_);	//Need to specify device 1[0] of platform 3[2] for dedicated graphics - Harri Laptop.
					if (errorStatus_)
						std::cout << "ERROR creating command queue for device. Status code: " << errorStatus_ << std::endl;

					std::cout << "\t\tDevice Name Chosen: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

					return;
				}
			}
			std::cout << std::endl;
		}
	}
	void initBuffersCL()
	{
		//Create input and output buffer for grid points//
		idGrid_ = cl::Buffer(context_, CL_MEM_READ_WRITE, gridByteSize_);
		modelGrid_ = cl::Buffer(context_, CL_MEM_READ_WRITE, gridByteSize_ * 3);
		boundaryGridBuffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE, gridByteSize_);
		outputBuffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE, maxOutputs_ * output_.bufferSize_ * sizeof(float));
		excitationBuffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE, excitation_.bufferSize_ * sizeof(float));
		inputPositionBuffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE, gridByteSize_);
		outputPositionBuffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE, gridByteSize_);
		connectionsBuffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE, gridByteSize_);
		coeffBuffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE, maxCoeffs_ * sizeof(float));

		//Copy data to newly created device's memory//
		float* temporaryGrid =  new float[gridElements_ * 3];
		memset(temporaryGrid, 0, gridByteSize_ * 3);

		//temporaryGrid[130808 * 1] = 1.0;
		//temporaryGrid[130808 * 2] = 1.0;

		commandQueue_.enqueueWriteBuffer(idGrid_, CL_TRUE, 0, gridByteSize_ , idGridInput_);
		commandQueue_.enqueueWriteBuffer(modelGrid_, CL_TRUE, 0, gridByteSize_*3, temporaryGrid);
		commandQueue_.enqueueWriteBuffer(boundaryGridBuffer_, CL_TRUE, 0, gridByteSize_, boundaryGridInput_);
		commandQueue_.enqueueWriteBuffer(inputPositionBuffer_, CL_TRUE, 0, gridByteSize_, inputGridInput_);
		commandQueue_.enqueueWriteBuffer(outputPositionBuffer_, CL_TRUE, 0, gridByteSize_, outputGridInput_);
		commandQueue_.enqueueWriteBuffer(connectionsBuffer_, CL_TRUE, 0, gridByteSize_, connections_);
	}
	void step()
	{
		commandQueue_.enqueueNDRangeKernel(kernelScheme_, cl::NullRange/*globaloffset*/, globalws_, localws_, NULL);
		//commandQueue_.finish();
		commandQueue_.enqueueNDRangeKernel(kernelConnections_, cl::NullRange/*globaloffset*/, globalws_, localws_, NULL);
		//commandQueue_.finish();

		output_.bufferIndex_++;
		excitation_.bufferIndex_++;
		bufferRotationIndex_ = (bufferRotationIndex_ + 1) % 3;
	}
protected:
public:
	FDTD_Accelerated(Implementation aImplementation, uint32_t aSampleRate, float aGridSpacing) : 
		implementation_(aImplementation),
		sampleRate_(aSampleRate),
		modelWidth_(128),
		modelHeight_(128),
		//model_(64, 64, 0.5),
		bufferSize_(aSampleRate),	//@ToDo - Make these controllable.
		output_(bufferSize_),
		excitation_(bufferSize_)
	{
		listenerPosition_[0] = 16;
		listenerPosition_[1] = 16;
		excitationPosition_[0] = 32;
		excitationPosition_[1] = 32;

		memset(emptyBuffer_, 0, 44100 * sizeof(float));
	}

	~FDTD_Accelerated()
	{
		delete vis;
	}

	//void fillBufferCPU(float* input, float* output, uint32_t numSteps)
	//{
	//	for(uint32_t i = 0; i != width)
	//}

	void fillBuffer(float* input, float* output, uint32_t numSteps)
	{
		//Set buffer length.
		kernelScheme_.setArg(10, sizeof(int), &numSteps);

		//Load excitation samples into GPU//
		commandQueue_.enqueueWriteBuffer(excitationBuffer_, CL_TRUE, 0, numSteps * sizeof(float), input);
		kernelScheme_.setArg(5, sizeof(cl_mem), &excitationBuffer_);

		//Calculate buffer size of synthesizer output samples//
		for (unsigned int i = 0; i != numSteps; ++i)
		{
			input[i] = 0.0;
			//Increments kernel indices//
			kernelScheme_.setArg(4, sizeof(int), &output_.bufferIndex_);
			kernelScheme_.setArg(3, sizeof(int), &bufferRotationIndex_);
			kernelConnections_.setArg(3, sizeof(int), &bufferRotationIndex_);

			step();

			//renderSimulation();

			//std::this_thread::sleep_for(std::chrono::milliseconds(2));
		}

		output_.resetIndex();
		excitation_.resetIndex();

		float* totalOutputBuffer = new float[numSteps*outputCounter_];
		commandQueue_.enqueueReadBuffer(outputBuffer_, CL_TRUE, 0, outputCounter_ * numSteps * sizeof(float), totalOutputBuffer);
		commandQueue_.enqueueWriteBuffer(outputBuffer_, CL_TRUE, 0, numSteps * sizeof(float), emptyBuffer_);

		mixOutput(totalOutputBuffer, output, numSteps);

		delete totalOutputBuffer;
	}
	void renderSimulation()
	{
		commandQueue_.enqueueReadBuffer(modelGrid_, CL_TRUE, 0, gridByteSize_, renderGrid);
		render(renderGrid);
	}


	void createModel(const std::string aPath, float aBoundaryValue, uint32_t aInputPosition[2], uint32_t aOutputPosition[2])
	{
		// JOSN parsing.
		//Read json file into program object//
		std::ifstream ifs(aPath);
		json jsonFile = json::parse(ifs);
		//std::cout << j << std::endl;

		modelWidth_ = jsonFile["buffer"].size();
		modelHeight_ = jsonFile["buffer"][0].size();

		inputGridInput_ = new int[modelWidth_*modelHeight_];
		outputGridInput_ = new int[modelWidth_*modelHeight_];
		boundaryGridInput_ = new float[modelWidth_*modelHeight_];
		idGridInput_ = new int[modelWidth_*modelHeight_];
		for (int i = 0; i != modelWidth_; ++i)
		{
			for (int j = 0; j != modelHeight_; ++j)
			{
				idGridInput_[i*modelWidth_ + j] = jsonFile["buffer"][i][j];
				outputGridInput_[i*modelWidth_ + j] = 0;
			}
		}

		globalws_ = cl::NDRange(modelWidth_, modelHeight_);
		localws_ = cl::NDRange(32, 32);						//@ToDo - CHANGE TO OPTIMIZED GROUP SIZE.

		//deviceType_ = INTEGRATED;
		deviceType_ = NVIDIA;

		initOpenCL();
		initRender();

		model_ = new Model(modelWidth_, modelHeight_, aBoundaryValue);
		model_->setInputPosition(aInputPosition[0], aInputPosition[1]);

		ofstream fpCoords;
		std::string strCoordsFilenameInit = "coords";
		for (int n = 0; n != 15; ++n)
		{
			std::string strCoordsFilename = strCoordsFilenameInit;
			strCoordsFilename.append(std::to_string(n));
			strCoordsFilename.append(".txt");
			fpCoords.open(strCoordsFilename.c_str());
			for (int i = 1; i != (modelWidth_ - 1); ++i)
			{
				for (int j = 1; j != (modelHeight_ - 1); ++j)
				{
					if (idGridInput_[i*modelWidth_ + j] == n)
					{
						fpCoords << j << ", " << i << "\n";
						fpCoords << i * modelWidth_ + j << "\n";
					}
				}
			}
			fpCoords.close();
		}

		//std::string fModelCoords = "positions";
		//myfile.open(fModelCoords);

		//for (int i = 1; i != (modelWidth_ - 1); ++i)
		//{
		//	for (int j = 1; j != (modelHeight_ - 1); ++j)
		//	{
		//		myfile << j << ", " << i << "\n";
		//		myfile << i * modelWidth_ + j << "\n";
		//	}
		//}

		//@TODO - Temporary post-processing boundary calculation. Remove when added in SVG parser.
		int boundaryCount = 0;
		for (int i = 1; i != (modelWidth_ - 1); ++i)
		{
			for (int j = 1; j != (modelHeight_ - 1); ++j)
			{
				//@ToDo - Work out calcualting boundary grid correctly.
				//This way trying to manually treat string differently from others.
				//if (idGridInput_[i*modelWidth_ + j] == 3)
				//{
				//	int gridID = idGridInput_[i*modelWidth_ + j];

				//	boundaryCount = 0;
				//	if (idGridInput_[(i - 1)*modelWidth_ + j] == gridID)
				//		++boundaryCount;
				//	if (idGridInput_[(i + 1)*modelWidth_ + j] == gridID)
				//		++boundaryCount;
				//	if (idGridInput_[(i - 1)*modelWidth_ + j + 1] == gridID)
				//		++boundaryCount;
				//	if (idGridInput_[(i - 1)*modelWidth_ + j - 1] == gridID)
				//		++boundaryCount;
				//	if (idGridInput_[(i + 1)*modelWidth_ + j + 1] == gridID)
				//		++boundaryCount;
				//	if (idGridInput_[(i + 1)*modelWidth_ + j - 1] == gridID)
				//		++boundaryCount;
				//	if (idGridInput_[(i)*modelWidth_ + j - 1] == gridID)
				//		++boundaryCount;
				//	if (idGridInput_[(i)*modelWidth_ + j + 1] == gridID)
				//		++boundaryCount;

				//	if (boundaryCount == 1)
				//		boundaryGridInput_[i*modelWidth_ + j] = 1.0;
				//}
				//else
				//{
				//	if (idGridInput_[i*modelWidth_ + j] > 0 && (idGridInput_[(i-1)*modelWidth_ + j] == 0 || idGridInput_[(i+1)*modelWidth_ + j] == 0 || idGridInput_[i*modelWidth_ + j-1] == 0 || idGridInput_[i*modelWidth_ + j+1] == 0))
				//	//if (idGridInput_[i][j] > 0 && (idGridInput_[i-1][j] == 0 || idGridInput_[i+1][j] == 0 || idGridInput_[i][j-1] == 0 || idGridInput_[i][j+1] == 0))
				//	{
				//		boundaryGridInput_[i*modelWidth_ + j] = 1.0;
				//	}
				//}
				/*if (idGridInput_[i*modelWidth_ + j] > 0)
				{
					boundaryCount = 0;
					if (idGridInput_[(i - 1)*modelWidth_ + j] == 0)
						++boundaryCount;
					if (idGridInput_[(i + 1)*modelWidth_ + j] == 0)
						++boundaryCount;
					if (idGridInput_[(i - 1)*modelWidth_ + j+1] == 0)
						++boundaryCount;
					if (idGridInput_[(i - 1)*modelWidth_ + j-1] == 0)
						++boundaryCount;
					if (idGridInput_[(i + 1)*modelWidth_ + j+1] == 0)
						++boundaryCount;
					if (idGridInput_[(i + 1)*modelWidth_ + j-1] == 0)
						++boundaryCount;
					if (idGridInput_[(i)*modelWidth_ + j-1] == 0)
						++boundaryCount;
					if (idGridInput_[(i)*modelWidth_ + j + 1] == 0)
						++boundaryCount;
					if(boundaryCount > 1 && boundaryCount < 5)
						boundaryGridInput_[i*modelWidth_ + j] = 1.0;
				}*/

				if (idGridInput_[i*modelWidth_ + j] > 0 && (idGridInput_[(i - 1)*modelWidth_ + j] == 0 || idGridInput_[(i + 1)*modelWidth_ + j] == 0 || idGridInput_[i*modelWidth_ + j - 1] == 0 || idGridInput_[i*modelWidth_ + j + 1] == 0))
					//if (idGridInput_[i][j] > 0 && (idGridInput_[i-1][j] == 0 || idGridInput_[i+1][j] == 0 || idGridInput_[i][j-1] == 0 || idGridInput_[i][j+1] == 0))
				{
					//boundaryGridInput_[i*modelWidth_ + j] = aBoundaryValue;
				}
				//std::cout << model_->boundaryGrid_.valueAt(i, j) << " | ";
			}
			//std::cout << std::endl;
		}
		//boundaryGridInput_[122376] = 1.0;
		//boundaryGridInput_[122888] = 1.0;
		//boundaryGridInput_[173064] = 1.0;
		//boundaryGridInput_[173576] = 1.0;

		//C3 for string 1 - Boundary at 232-162=70.
		boundaryGridInput_[17668] = 1.0;
		boundaryGridInput_[17924] = 1.0;
		//C#3 for string 2 - Boundary at 229-154=75.
		boundaryGridInput_[18955] = 1.0;
		boundaryGridInput_[19211] = 1.0;
		//D3 for string 3 - Boundary at 227-146=81.
		boundaryGridInput_[20498] = 1.0;
		boundaryGridInput_[20754] = 1.0;
		//D#3 for string 4 - Boundary at 226-138=88
		boundaryGridInput_[22297] = 1.0;
		boundaryGridInput_[22553] = 1.0;
		//E3 for string 5 - Boundary at 230-131=99
		boundaryGridInput_[25120] = 1.0;
		boundaryGridInput_[25376] = 1.0;
		//F3 for string 6 - Boundary at 229-124=105
		boundaryGridInput_[26664] = 1.0;
		boundaryGridInput_[26920] = 1.0;
		//F#3 for string 7 - Boundary at 231-117=114
		boundaryGridInput_[28974] = 1.0;
		boundaryGridInput_[29230] = 1.0;
		//G3 for string 8 - Boundary at 226-111=115
		boundaryGridInput_[29238] = 1.0;
		boundaryGridInput_[29494] = 1.0;
		//G#3 for string 9 - Boundary at 227-105=122
		boundaryGridInput_[31037] = 1.0;
		boundaryGridInput_[31293] = 1.0;
		//A3 for string 10 - Boundary at 230-99=131
		boundaryGridInput_[33347] = 1.0;
		boundaryGridInput_[33603] = 1.0;
		//A#3 for string 10 - Boundary at 230-99=131
		boundaryGridInput_[33347] = 1.0;
		boundaryGridInput_[33603] = 1.0;
		//A#3 for string 11 - Boundary at 233-94=139
		boundaryGridInput_[35402] = 1.0;
		boundaryGridInput_[35658] = 1.0;
		//B3 for string 12 - Boundary at230-88=142
		boundaryGridInput_[36177] = 1.0;
		boundaryGridInput_[36433] = 1.0;
		//C4 for string 13 - Boundary at 227-83=144
		boundaryGridInput_[36441] = 1.0;
		boundaryGridInput_[36697] = 1.0;

		gridElements_ = (modelWidth_ * modelHeight_);
		gridByteSize_ = (gridElements_ * sizeof(float));
		renderGrid = new float[gridElements_ * 3];
		connections_ = new int[gridElements_]();

		//Load up connections - @ToDo - Make this automatically loaded from json configration of model rather than manual.

		//Grid way

		//256
		connections_[29078] = 59140;
		connections_[29083] = 59403;
		//connections_[225310] = 112519;
		//connections_[230953] = 112424;
		//connections_[231989] = 112350;
		//connections_[231487] = 112296;
		//connections_[232521] = 111974;
		//connections_[232020] = 111884;
		//connections_[233569] = 111812;
		//connections_[233067] = 111477;
		//connections_[233591] = 111392;
		//connections_[234114] = 111310;
		//connections_[234638] = 110985;

		//512
		//connections_[233992] = 112908;
		//connections_[236051] = 112899;
		//connections_[225310] = 112519;
		//connections_[230953] = 112424;
		//connections_[231989] = 112350;
		//connections_[231487] = 112296;
		//connections_[232521] = 111974;
		//connections_[232020] = 111884;
		//connections_[233569] = 111812;
		//connections_[233067] = 111477;
		//connections_[233591] = 111392;
		//connections_[234114] = 111310;
		//connections_[234638] = 110985;

		if (implementation_ == Implementation::OPENCL)
		{
			initBuffersCL();
		}

		createExplicitEquation(aPath);
	}

	void createExplicitEquation(const std::string aPath)
	{
		//Read json file into program object//
		std::ifstream ifs(aPath);
		json jsonFile = json::parse(ifs);
		std::string sourceFile = jsonFile["controllers"][0]["physics_kernel"];

		//Create program source object from std::string source code//
		std::vector<std::string> programSources;
		programSources.push_back(sourceFile);
		cl::Program::Sources source(programSources);	//Apparently this takes a vector of strings as the program source.

		//Create program from source code//
		kernelProgram_ = cl::Program(context_, source, &errorStatus_);
		if (errorStatus_)
			std::cout << "ERROR creating OpenCL program from source. Status code: " << errorStatus_ << std::endl;
		
		//Build program//
		char options[1024];
		snprintf(options, sizeof(options),
			" -cl-fast-relaxed-math"
			" -cl-single-precision-constant"
			//""
		);
		kernelProgram_.build();	//@Highlight - Keep this in?

		kernelScheme_ = cl::Kernel(kernelProgram_, "fdtdKernel", &errorStatus_);	//@ToDo - Hard coded the kernel name. Find way to generate this?
		kernelConnections_ = cl::Kernel(kernelProgram_, "connectionsKernel", &errorStatus_);	//@ToDo - Hard coded the kernel name. Find way to generate this?
		if (errorStatus_)
		{
			std::cout << "ERROR building OpenCL kernel from source. Status code: " << errorStatus_ << std::endl;

			//juce::Logger::getCurrentLogger()->outputDebugString("ERROR building OpenCL kernel from source.Status code : ");
		}

		kernelScheme_.setArg(0, sizeof(cl_mem), &idGrid_);
		kernelScheme_.setArg(1, sizeof(cl_mem), &modelGrid_);
		kernelScheme_.setArg(2, sizeof(cl_mem), &boundaryGridBuffer_);
		kernelScheme_.setArg(6, sizeof(cl_mem), &outputBuffer_);

		kernelConnections_.setArg(0, sizeof(cl_mem), &idGrid_);
		kernelConnections_.setArg(1, sizeof(cl_mem), &modelGrid_);
		kernelConnections_.setArg(2, sizeof(cl_mem), &boundaryGridBuffer_);
		kernelConnections_.setArg(4, sizeof(int), &numConnections_);
		kernelConnections_.setArg(5, sizeof(cl_mem), &connectionsBuffer_);

		int inPos = model_->getInputPosition();
		int outPos = model_->getOutputPosition();
		kernelScheme_.setArg(7, sizeof(cl_mem), &inputPositionBuffer_);
		kernelScheme_.setArg(8, sizeof(cl_mem), &outputPositionBuffer_);
	}
	void createMatrixEquation(const std::string aPath);	//How is the matrix equations defined? Is there just a default matrix equation that can be formed for many equations or need be defined?

	//@ToDo - Do we need this? Coefficients just need to use .setArg(), don't need to create buffer for them...
	void createCoefficient(std::string aCoeff)
	{

	}
	void updateCoefficient(std::string aCoeff, uint32_t aIndex, float aValue)
	{
		kernelScheme_.setArg(aIndex, sizeof(float), &aValue);	//@ToDo - Need dynamicaly find index for setArg (The first param)
	}
	void updateCoefficientConnection(std::string aCoeff, uint32_t aIndex, float aValue)
	{
		kernelConnections_.setArg(aIndex, sizeof(float), &aValue);	//@ToDo - Need dynamicaly find index for setArg (The first param)
	}

	void updateCoefficient(std::string aCoeff, uint32_t aIndex, double aValue)
	{
		kernelScheme_.setArg(aIndex, sizeof(double), &aValue);	//@ToDo - Need dynamicaly find index for setArg (The first param)
	}

	void resetInputPosition()
	{
		memset(inputGridInput_, 0, gridByteSize_);
		commandQueue_.enqueueWriteBuffer(inputPositionBuffer_, CL_TRUE, 0, gridByteSize_, inputGridInput_);
		kernelScheme_.setArg(7, sizeof(cl_mem), &inputPositionBuffer_);
	}
	void setInputPosition(int aInputs[])
	{
		model_->setInputPosition(aInputs[0], aInputs[1]);
		int inPos = (aInputs[1] * 256 + aInputs[0]);
		inputGridInput_[inPos] = 1;
	}
	void updateInputPositions()
	{
		commandQueue_.enqueueWriteBuffer(inputPositionBuffer_, CL_TRUE, 0, gridByteSize_, inputGridInput_);
		kernelScheme_.setArg(7, sizeof(cl_mem), &inputPositionBuffer_);
	}
	void mixOutput(float* aInput, float* aOutput, int bufferSize)
	{
		for (uint32_t i = 0; i != bufferSize; ++i)
		{
			float sum = 0.0;
			for (uint32_t j = 1; j != outputCounter_; ++j)
				sum += aInput[(j-1)*bufferSize + i];
			aOutput[i] = sum / (float)(outputCounter_+1.0);
		}
	}
	void setOutputPosition(int aOutputs[])
	{
		model_->setOutputPosition(aOutputs[0], aOutputs[1]);
		int flatPosition = model_->getOutputPosition();
		outputGridInput_[flatPosition] = outputCounter_++;
		commandQueue_.enqueueWriteBuffer(outputPositionBuffer_, CL_TRUE, 0, gridByteSize_, outputGridInput_);
		kernelScheme_.setArg(8, sizeof(cl_mem), &outputPositionBuffer_);
	}
	void setInputPositions(std::vector<uint32_t> aInputs);
	void setOutputPositions(std::vector<uint32_t> aOutputs);

	void setInputs(std::vector<std::vector<float>> aInputs);

	std::vector<std::vector<float>> getInputs();
	std::vector<std::vector<float>> getOutputs();

	void initRender()
	{
		vis = new Visualizer(modelWidth_, modelHeight_);
	}
	void render(float* aData)
	{
		vis->render(aData);
	}
	GLFWwindow* getWindow()
	{
		return vis->getWindow();
	}

	int getModelWidth()
	{
		return modelWidth_;
	}
	int getModelHeight()
	{
		return modelHeight_;
	}

	void resetState()
	{
		//Copy data to newly created device's memory//
		float* temporaryGrid = new float[gridElements_ * 3];
		memset(temporaryGrid, 0, gridByteSize_ * 3);

		commandQueue_.enqueueWriteBuffer(modelGrid_, CL_TRUE, 0, gridByteSize_ * 3, temporaryGrid);

		kernelScheme_.setArg(1, sizeof(cl_mem), &modelGrid_);
		kernelConnections_.setArg(1, sizeof(cl_mem), &modelGrid_);
	}

	void getModels(int aNumPointsStrings[], int aNumPointsPlate, float** aStrings, float* aPlate)
	{
		commandQueue_.enqueueReadBuffer(modelGrid_, CL_TRUE, 0, gridByteSize_, renderGrid);

		aNumPointsPlate = 0;
		for (int i = 0; i != (modelWidth_); ++i)
		{
			for (int j = 0; j != (modelHeight_); ++j)
			{
				if (idGridInput_[i*modelWidth_ + j] == 1)
				{
					aPlate[aNumPointsPlate++] = renderGrid[i*modelWidth_ + j];
				}
			}
		}
		for (int n = 0; n != 13; ++n)
		{
			aNumPointsStrings[n] = 0;
			for (int i = 1; i != (modelWidth_ - 1); ++i)
			{
				for (int j = 1; j != (modelHeight_ - 1); ++j)
				{
					if (idGridInput_[i*modelWidth_ + j] == n+2)
					{
						aStrings[n][aNumPointsStrings[n]++] = renderGrid[i*modelWidth_ + j];
					}
				}
			}
		}
	}

	void updateCoefficients(float aCoeffs[], uint32_t aNumCoeffs)
	{
		commandQueue_.enqueueWriteBuffer(coeffBuffer_, CL_TRUE, 0, aNumCoeffs * sizeof(float), aCoeffs);
		kernelScheme_.setArg(9, sizeof(cl_mem), &coeffBuffer_);
	}
};

#endif