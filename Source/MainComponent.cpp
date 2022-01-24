#include "MainComponent.h"

#include <tuple>

float secondInput[44100];
double xposInput;
double yposInput;
int inputPos[2]{ 0, 0 };
int outputPos[2]{ 0, 0 };

#define M_PI 3.14159265358979323846264338327950288

StringExciter::StringExciter()
{
	Fe.resize(maxLength);

	for (int i = 0; i < exciterLength; i++)
		Fe[i] = Fmax / 2.0f * (1 - cos(q * M_PI * i / exciterLength));
}

void StringExciter::excite()
{
	play = true;
}
void StringExciter::setLength(int L)
{
	if (L > maxLength)
		L = maxLength;

	exciterLength = L;

	for (int i = 0; i < exciterLength; i++)
		Fe[i] = Fmax * 0.5f * (1.0f - cos(q * M_PI * i / exciterLength));
}

void StringExciter::setLevel(double level)
{
	for (int i = 0; i < exciterLength; i++)
	{
		Fe[i] = Fmax * 0.5f * (1.0f - cos(q * M_PI * i / exciterLength));
	}
}

double StringExciter::getOutput()
{
	double output = 0.0f;

	if (play)
	{
		output = Fe[pos];
		pos++;
	}

	if (pos >= exciterLength)
	{
		pos = 0;
		play = false;
	}
	return output;
}

//==============================================================================
MainComponent::MainComponent() : wavetableExciter_(1, &(wave[0]), wave.size()),
									audioSetupComp(deviceManager,
										0,     // minimum input channels
										256,   // maximum input channels
										0,     // minimum output channels
										256,   // maximum output channels
										false, // ability to select midi inputs
										false, // ability to select midi output device
										false, // treat channels as stereo pairs
										false) // hide advanced options
{
	addAndMakeVisible(audioSetupComp);
	//addAndMakeVisible(diagnosticsBox);

	diagnosticsBox.setMultiLine(true);
	diagnosticsBox.setReturnKeyStartsNewLine(true);
	diagnosticsBox.setReadOnly(true);
	diagnosticsBox.setScrollbarsShown(true);
	diagnosticsBox.setCaretVisible(false);
	diagnosticsBox.setPopupMenuEnabled(true);
	diagnosticsBox.setColour(juce::TextEditor::backgroundColourId, juce::Colour(0x32ffffff));
	diagnosticsBox.setColour(juce::TextEditor::outlineColourId, juce::Colour(0x1c000000));
	diagnosticsBox.setColour(juce::TextEditor::shadowColourId, juce::Colour(0x16000000));

    // Make sure you set the size of the component after
    // you add any child components.
    setSize (800, 600);

    // Some platforms require permissions to open input channels so request that here
    if (juce::RuntimePermissions::isRequired (juce::RuntimePermissions::recordAudio)
        && ! juce::RuntimePermissions::isGranted (juce::RuntimePermissions::recordAudio))
    {
        juce::RuntimePermissions::request (juce::RuntimePermissions::recordAudio,
                                           [&] (bool granted) { setAudioChannels (granted ? 2 : 0, 2); });
    }
    else
    {
        // Specify the number of input and output channels that we want to open
        setAudioChannels (2, 2);
    }

	//Interface//
	addAndMakeVisible(sldPropagationOne);
	sldPropagationOne.setRange(0.000001, 1000, 0.0000001);
	sldPropagationOne.setValue(50);
	sldPropagationOne.addListener(this);
	addAndMakeVisible(sldDampingOne);
	sldDampingOne.setRange(0.0, 6.9999, 0.000001);
	sldDampingOne.setValue(0.1);
	sldDampingOne.addListener(this);

	addAndMakeVisible(sldPropagationTwo);
	sldPropagationTwo.setRange(0.0, 0.49, 0.0001);
	sldPropagationTwo.setValue(1000.0);
	sldPropagationTwo.addListener(this);
	addAndMakeVisible(sldDampingTwo);
	sldDampingTwo.setRange(0.0, 1.0, 0.000001);
	sldDampingTwo.setValue(0.005);
	sldDampingTwo.addListener(this);

	addAndMakeVisible(sldPropagationPlate);
	sldPropagationPlate.setRange(0.000001, 1000, 0.0000001);
	sldPropagationPlate.setValue(50);
	sldPropagationPlate.addListener(this);
	addAndMakeVisible(sldDampingPlateOne);
	sldDampingPlateOne.setRange(0.0, 6.9999, 0.000001);
	sldDampingPlateOne.setValue(0.1);
	sldDampingPlateOne.addListener(this);
	addAndMakeVisible(sldDampingPlateTwo);
	sldDampingPlateTwo.setRange(0.0, 1.0, 0.000001);
	sldDampingPlateTwo.setValue(0.005);
	sldDampingPlateTwo.addListener(this);

	addAndMakeVisible(sldInputDuration);
	sldInputDuration.setRange(0.0, 1000, 1.0);
	sldInputDuration.setValue(10);
	sldInputDuration.addListener(this);

	//Labels//
	addAndMakeVisible(lblPropagationOne);
	lblPropagationOne.setText("Kappa", dontSendNotification);
	lblPropagationOne.attachToComponent(&sldPropagationOne, true);
	addAndMakeVisible(lblDampingOne);
	lblDampingOne.setText("General Damping", dontSendNotification);
	lblDampingOne.attachToComponent(&sldDampingOne, true);
	addAndMakeVisible(lblPropagationTwo);
	lblPropagationTwo.setText("Frequency Damping", dontSendNotification);
	lblPropagationTwo.attachToComponent(&sldPropagationTwo, true);
	addAndMakeVisible(lblDampingTwo);
	lblDampingTwo.setText("Damping Two", dontSendNotification);
	lblDampingTwo.attachToComponent(&sldDampingTwo, true);
	addAndMakeVisible(lblInputDuration);
	lblInputDuration.setText("Input Duration", dontSendNotification);
	lblInputDuration.attachToComponent(&sldInputDuration, true);

	addAndMakeVisible(lblFPS);
	lblFPS.setText("FPS: ", dontSendNotification);
	addAndMakeVisible(lblLatency);
	lblLatency.setText("Latency: ", dontSendNotification);

	mutexInit.lock();
	Implementation impl = OPENCL;
	unsigned int bufferFrames = 1024; // 256 sample frames
	const double gridSpacing = 0.001;
	simulationModel = new FDTD_Accelerated(impl, bufferFrames, gridSpacing);
	uint32_t inputPosition[2] = { 0, 0 };
	uint32_t outputPosition[2] = { 0, 0 };
	float boundaryValue = 1.0;
	simulationModel->createModel(physicalModelPath_, boundaryValue, inputPosition, outputPosition);

	// Update Coefficients.
	// 512
	_strKappa = 50;
	_strSigmaZero = 1.000;
	_strSigmaOne = 0.005;
	// 64
	//double stringFundamentalFrequency = 210;
	//double stringGenDamp = 1.000;
	//double stringFreqDamp = 0.005;
	//setStringCoeffs(_strKappa, _strSigmaZero, _strSigmaOne);
	setDefaultStringCoeffs(_strKappa, _strSigmaZero, _strSigmaOne);

	//512
	 _plateKappa = 100;
	 _plateSigmaZero = 1.945476;
	 _plateSigmaOne = 0.00005;
	//64
	//double plateFundamentalFrequency = 1;
	//double plateGenDamp = 0.1;
	//double plateFreqDamp = 0.005;
	setPlateCoeffs(_plateKappa, _plateSigmaZero, _plateSigmaOne);

	//Update buffer of Coefficients.
	simulationModel->updateCoefficients(coeffs, numCoeffs_);

	//Setup output position
	// String - 512
	//outputPos[0] = 7;	//use_case_512
	//outputPos[1] = 464;

	//use_case_256_13
	for (uint32_t i = 0; i != 13; ++i)
	{
		outputPos[0] = outputCoords[i];
		outputPos[1] = 214;
		simulationModel->setOutputPosition(outputPos);
	}

	//use_case_512_13
	//for (uint32_t i = 0; i != 13; ++i)
	//{
	//	outputPos[0] = outputCoords[i];
	//	outputPos[1] = 233;
	//	simulationModel->setOutputPosition(outputPos);
	//}

	// String - 64
	//outputPos[0] = 4;
	//outputPos[1] = 51;
	//simulationModel->setOutputPosition(outputPos);
	//outputPos[0] = 10;
	//outputPos[1] = 40;
	outputPos[0] = simulationModel->getModelHeight() / 2.0;
	outputPos[1] = simulationModel->getModelWidth() / 2.0;
	simulationModel->setOutputPosition(outputPos);
	//outputPos[0] = simulationModel->getModelHeight() / 2.0;
	//outputPos[1] = (simulationModel->getModelWidth() / 4.0) * 2.5;
	//simulationModel->setOutputPosition(outputPos);

	//Sensel Init.
	senselInterfaceOne = new Sensel(1);
	senselInterfaceTwo = new Sensel(0);

	//Timer callback
	startTimer(1);
	isInit = true;

	for (uint32_t i = 0; i != 13; ++i)
	{
		uStrings[i] = new float[99999];
		memset(uStrings[i], 0, 99999 * sizeof(float));
	}
	uPlate = new float[999999];
	memset(uPlate, 0, 999999 * sizeof(float));

	connectionPointsIdx.push_back(std::make_tuple(151, 113, 214));	//112908, 233992
	connectionPointsIdx.push_back(std::make_tuple(156, 113, 214));	//, 235539
	connectionPointsIdx.push_back(std::make_tuple(151, 113, 214));
	connectionPointsIdx.push_back(std::make_tuple(151, 113, 214));
	connectionPointsIdx.push_back(std::make_tuple(151, 113, 214));
	connectionPointsIdx.push_back(std::make_tuple(151, 113, 214));
	connectionPointsIdx.push_back(std::make_tuple(151, 113, 214));
	connectionPointsIdx.push_back(std::make_tuple(151, 113, 214));
	connectionPointsIdx.push_back(std::make_tuple(151, 113, 214));
	connectionPointsIdx.push_back(std::make_tuple(151, 113, 214));
	connectionPointsIdx.push_back(std::make_tuple(151, 113, 214-4));
	connectionPointsIdx.push_back(std::make_tuple(151, 113, 214-5));

	connectionPointsCoords.push_back(std::make_tuple(0, 0, 0, 0));
	connectionPointsCoords.push_back(std::make_tuple(0, 0, 0, 0));
	connectionPointsCoords.push_back(std::make_tuple(0, 0, 0, 0));
	connectionPointsCoords.push_back(std::make_tuple(0, 0, 0, 0));
	connectionPointsCoords.push_back(std::make_tuple(0, 0, 0, 0));
	connectionPointsCoords.push_back(std::make_tuple(0, 0, 0, 0));
	connectionPointsCoords.push_back(std::make_tuple(0, 0, 0, 0));
	connectionPointsCoords.push_back(std::make_tuple(0, 0, 0, 0));
	connectionPointsCoords.push_back(std::make_tuple(0, 0, 0, 0));
	connectionPointsCoords.push_back(std::make_tuple(0, 0, 0, 0));
	connectionPointsCoords.push_back(std::make_tuple(0, 0, 0, 0));
	connectionPointsCoords.push_back(std::make_tuple(0, 0, 0, 0));

	stringExciter_.setQ(2);
	stringExciter_.excite();
	stringExciter_.setLength(10);

	simulationModel->getModels(numPointsStrings, numPointsPlate, uStrings, uPlate);


	audioOutput_.open("audiooutput.raw", std::ios::binary);

	filter = new IIRFilter();
	filter->setCoefficients(IIRCoefficients::makeHighPass(44100, 150, 1.0));

	mutexInit.unlock();
}

MainComponent::~MainComponent()
{
    // This shuts down the audio device and clears the audio source.
    shutdownAudio();
	audioOutput_.close();
}
//==============================================================================
void MainComponent::prepareToPlay (int samplesPerBlockExpected, double sampleRate)
{
    // This function will be called when the audio device is started, or when
    // its settings (i.e. sample rate, block size, etc) are changed.

    // You can use this function to initialise any resources you might need,
    // but be careful - it will be called on the audio thread, not the GUI thread.

    // For more details, see the help for AudioProcessor::prepareToPlay()
}

double clamp(double in, double min, double max)
{
	if (in > max)
		return max;
	else if (in < min)
		return min;
	else
		return in;
}
void MainComponent::setStringCoeffs(double aFundFreq, double aSigmaOne, double aSigmaTwo)
{
	double maxGamma = 2000;
	double gamma = aFundFreq * 2;				// Wave speed = c/2L
	double sampleRate = 44100;
	double k = 1.0f / sampleRate;				//Timestep.

	double s0 = aSigmaOne;						// Frequency-independent damping
	double s1 = aSigmaTwo;						// Frequency-dependent damping

	// Courant numbers
	double N = 478;								//478 // @TODO - Needs to be the number of grid points making up string. Grid is 512, but string is less. How to calculate this??
	double kappa = 0.02;
	double h = 1.0 / N;

	h = sqrt((maxGamma * maxGamma * k * k + 4.0 * s1 * k + sqrt(pow(maxGamma * maxGamma * k * k + 4.0 * s1 * k, 2.0) + 16.0 * kappa * kappa * k * k)) * 0.5);
	//h = sqrt((gamma * gamma * k * k + 4.0 * s1 * k + sqrt(pow(gamma * gamma * k * k + 4.0 * s1 * k, 2.0) + 16.0 * kappa * kappa * k * k)) * 0.5);
	//N = floor(1.0 / h); // Number of gridpoints

	double timeCond = (h*h) / (2 * kappa);
	

	double lambda = gamma * k / h;
	double lambdaSq = lambda * lambda;
	double mu = k * kappa / (h * h);
	double muSq = mu * mu;

	double gridMin = sqrt((gamma*gamma*k*k + sqrt(gamma*gamma*gamma*gamma*k*k*k*k + 16 * kappa * kappa * k * k)) / 2);

	double B1 = s0 * k;
	double B2 = (2 * s1 * k) / (h * h);

	double b1 = 2.0 / (k * k);
	double b2 = (2 * s1) / (k * h * h);

	float D = (1.0 + s0 * k);

	float A1 = 2 - 2 * lambdaSq - 6 * muSq - 2 * B2;
	float A2 = lambdaSq + 4 * muSq + B2;
	float A3 = muSq;
	float A4 = B1 - 1 + 2 * B2;
	float A5 = B2;

	//A1 *= D;
	//A2 *= D;
	//A3 *= D;
	//A4 *= D;
	//A5 *= D;
	/*simulationModel->updateCoefficient("strLambdaOne", 15, A1);
	simulationModel->updateCoefficient("strLambdaFive", 16, A5);
	simulationModel->updateCoefficient("strLambdaTwo", 17, A2);
	simulationModel->updateCoefficient("strLambdaThree", 18, A3);
	simulationModel->updateCoefficient("strLambdaFour", 19, A4);*/

	simulationModel->updateCoefficient("strOneCoeffOne", 18, A1);
	simulationModel->updateCoefficient("strOneCoeffFive", 16, A5);
	simulationModel->updateCoefficient("strOneCoeffTwo", 15, A2);
	simulationModel->updateCoefficient("strOneCoeffThree", 19, A3);
	simulationModel->updateCoefficient("strOneCoeffFour", 17, A4);
	simulationModel->updateCoefficient("strDamp", 21, D);

	simulationModel->updateCoefficientConnection("strDeltaH", 7, (float)h);
	simulationModel->updateCoefficientConnection("strDeltaH", 8, (float)k);
}
void MainComponent::setDefaultStringCoeffs(double aFundFreq, double aSigmaOne, double aSigmaTwo)
{
	for (uint32_t i = 0; i != 13; ++i)
	{
		double fundFreq = fundamentalFrequencies[i];

		double sigmaZero = aSigmaOne;
		double SigmaOne = aSigmaTwo;

		double maxGamma = (32 + 13 * aFundFreq)*2;
		double gamma = fundFreq * 2;				// Wave speed = c/2L
		double sampleRate = 44100;
		double k = 1.0f / sampleRate;				//Timestep.

		double s0 = sigmaZero;						// Frequency-independent damping
		double s1 = SigmaOne;						// Frequency-dependent damping

		// Courant numbers
		double N = 278;								//478 // @TODO - Needs to be the number of grid points making up string. Grid is 512, but string is less. How to calculate this??
		double kappa = 0.2;
		double h = 1.0 / N;

		//h = sqrt((maxGamma * maxGamma * k * k + 4.0 * s1 * k + sqrt(pow(maxGamma * maxGamma * k * k + 4.0 * s1 * k, 2.0) + 16.0 * kappa * kappa * k * k)) * 0.5);
		h = sqrt((gamma * gamma * k * k + 4.0 * s1 * k + sqrt(pow(gamma * gamma * k * k + 4.0 * s1 * k, 2.0) + 16.0 * kappa * kappa * k * k)) * 0.5);
		//N = floor(1.0 / h); // Number of gridpoints

		int realN = 1 / h;

		double timeCond = (h*h) / (2 * kappa);


		double lambda = gamma * k / h;
		double lambdaSq = lambda * lambda;
		double mu = k * kappa / (h * h);
		double muSq = mu * mu;

		double gridMin = sqrt((gamma*gamma*k*k + sqrt(gamma*gamma*gamma*gamma*k*k*k*k + 16 * kappa * kappa * k * k)) / 2);

		double B1 = s0 * k;
		double B2 = (2 * s1 * k) / (h * h);

		double b1 = 2.0 / (k * k);
		double b2 = (2 * s1) / (k * h * h);

		float D = (1.0 + s0 * k);

		float A1 = 2 - 2 * lambdaSq - 6 * muSq - 2 * B2;
		float A2 = lambdaSq + 4 * muSq + B2;
		float A3 = muSq;
		float A4 = B1 - 1 + 2 * B2;
		float A5 = B2;

		//A1 *= D;
		//A2 *= D;
		//A3 *= D;
		//A4 *= D;
		//A5 *= D;
		/*simulationModel->updateCoefficient("strLambdaOne", 15, A1);
		simulationModel->updateCoefficient("strLambdaFive", 16, A5);
		simulationModel->updateCoefficient("strLambdaTwo", 17, A2);
		simulationModel->updateCoefficient("strLambdaThree", 18, A3);
		simulationModel->updateCoefficient("strLambdaFour", 19, A4);*/

		//simulationModel->updateCoefficient("strOneCoeffOne", 18, A1);
		//simulationModel->updateCoefficient("strOneCoeffFive", 16, A5);
		//simulationModel->updateCoefficient("strOneCoeffTwo", 15, A2);
		//simulationModel->updateCoefficient("strOneCoeffThree", 19, A3);
		//simulationModel->updateCoefficient("strOneCoeffFour", 17, A4);
		//simulationModel->updateCoefficient("strDamp", 21, D);

		coeffs[7 + i*6 + 0] = A1;
		coeffs[7 + i*6 + 1] = A2;
		coeffs[7 + i*6 + 2] = A3;
		coeffs[7 + i*6 + 3] = A4;
		coeffs[7 + i*6 + 4] = A5;
		coeffs[7 + i*6 + 5] = D;

		simulationModel->updateCoefficientConnection("strDeltaH", 7, (float)h);
		simulationModel->updateCoefficientConnection("strDeltaH", 8, (float)k);
	}
}
void MainComponent::setPlateCoeffs(double aKappa, double aSigmaZero, double aSigmaOne)
{
	_plateKappa = aKappa;
	_plateSigmaZero = aSigmaZero;
	_plateSigmaOne = aSigmaOne;

	//double maxKappaSq = 25000;	//512
	double maxSigmaOne = 1.0;
	double maxKappaSq = 1000 * 1000;		//64
	double kappaSq = clamp(_plateKappa*_plateKappa, 0.1, maxKappaSq);
	double sampleRate = 44100;
	double k = 1.0f / sampleRate;			//Timestep.

	//double h = 1.1*(2 * sqrt(k*(maxSigmaOne * maxSigmaOne + sqrt(maxKappaSq + maxSigmaOne * maxSigmaOne))));
	double h = 2*sqrt(k*(_plateSigmaOne * _plateSigmaOne + sqrt(kappaSq + _plateSigmaOne * _plateSigmaOne)));
	//double h = 0.0095;
	//double h = 0.30117005389;
	//double h = 1.0 / 512.0;

	//Connection information
	float poisson_ratio = 0.3;
	float youngs = 200000000000;
	float dValue = 0.0;
	float densityTimesThickness = dValue / (_plateKappa * _plateKappa);

	float mu = (k * _plateKappa) / (h*h);
	float muSqr = mu * mu;
	float d = (1.0f + _plateSigmaZero * k);
	float S = (2.0f * _plateSigmaOne * k) / (h * h);							// S = 2 * sigmaOne * kappa / h^2

	float lambdaFour = muSqr;		// (mu^2)
	float lambdaThree =  2.0f * muSqr;
	float lambdaTwo = 8.0f * muSqr + S;								// (8 * mu^2)
	float lambdaOne = (2.0f - 20.0f * (mu*mu) - 4.0f * S );
	float lambdaFive = (_plateSigmaZero * k - 1.0f + 4.0f * S);			//(sigmaZero * k - 1 + 4 * S)
	float lambdaSix = S;
	float C4 = (k * k);

	/*simulationModel->updateCoefficient("lambdaFive", 9, lambdaFive);
	simulationModel->updateCoefficient("lambdaTwo", 10, lambdaTwo);
	simulationModel->updateCoefficient("lambdaFour", 11, lambdaFour);
	simulationModel->updateCoefficient("lambdaSix", 12, lambdaSix);
	simulationModel->updateCoefficient("lambdaThree", 13, lambdaThree);
	simulationModel->updateCoefficient("lambdaOne", 14, lambdaOne);
	simulationModel->updateCoefficient("damp", 20, d);*/

	//simulationModel->updateCoefficient("plateCoeffFive", 11, lambdaFive);
	//simulationModel->updateCoefficient("plateCoeffTwo", 10, lambdaTwo);
	//simulationModel->updateCoefficient("plateCoeffFour", 9, lambdaFour);
	//simulationModel->updateCoefficient("plateCoeffSix", 13, lambdaSix);
	//simulationModel->updateCoefficient("plateCoeffThree", 12, lambdaThree);
	//simulationModel->updateCoefficient("plateCoeffOne", 14, lambdaOne);
	//simulationModel->updateCoefficient("damp", 20, d);

	coeffs[0] = lambdaOne;
	coeffs[1] = lambdaTwo;
	coeffs[2] = lambdaThree;
	coeffs[3] = lambdaFour;
	coeffs[4] = lambdaFive;
	coeffs[5] = lambdaSix;
	coeffs[6] = d;

	simulationModel->updateCoefficientConnection("plateDeltaH", 6, (float)h);
}

int counter = 0;
void MainComponent::getNextAudioBlock (const juce::AudioSourceChannelInfo& bufferToFill)
{
	if (isInit)
	{
		mutexInit.lock();

		// Your audio-processing code goes here!
		auto level = 0.125f;
		auto* leftBuffer = bufferToFill.buffer->getWritePointer(0, bufferToFill.startSample);
		auto* rightBuffer = bufferToFill.buffer->getWritePointer(1, bufferToFill.startSample);

		//Input excitation//
		simulationModel->updateInputPositions();
		for (int j = 0; j != bufferToFill.numSamples; ++j)
		{
			//inputExcitation[j] = sineExciter_.getNextSample();
			if (wavetableExciter_.isExcitation())
			{
				inputExcitation[j] = wavetableExciter_.getNextSample() * 0.5;
				inputExcitation[j] = stringExciter_.getOutput()* 0.5;
				//inputExcitation[j] = bow_.sample() * 0.05;
			}
			else
			{
				inputExcitation[j] = 0;
			}
			//if (isExcite)
			//{
			//	inputExcitation[j] = 1.0;
			//	isExcite = false;
			//}
			//else
			//{
			//	inputExcitation[j] = 0.0;
			//}
		}

		// For more details, see the help for AudioProcessor::getNextAudioBlock()
		isPlayed_ = true;
		simulationModel->fillBuffer(inputExcitation, leftBuffer, bufferToFill.numSamples);
		//filter->processSamples(leftBuffer, bufferToFill.numSamples);						//Highpassfilter.
		memcpy(rightBuffer, leftBuffer, bufferToFill.numSamples * sizeof(float));

		audioOutput_.write((char*)(leftBuffer), bufferToFill.numSamples * sizeof(float));

		//for (uint32_t i = 0; i != bufferToFill.numSamples; ++i)
		//{
		//	//leftBuffer[i] = bow_.sample();
		//	
		//}
		//memcpy(rightBuffer, leftBuffer, bufferToFill.numSamples * sizeof(float));

		counter += (bufferToFill.numSamples);
		if (counter > framerate)
		{
			//simulationModel->renderSimulation();
			simulationModel->getModels(numPointsStrings, numPointsPlate, uStrings, uPlate);
			counter = 0;
			repaint();
		}

		if(isPlayed_)
			simulationModel->resetInputPosition();

		mutexInit.unlock();
	}
}

void MainComponent::releaseResources()
{
    // This will be called when the audio device stops, or when it is being
    // restarted due to a setting change.

    // For more details, see the help for AudioProcessor::releaseResources()
}
Path MainComponent::generateStringPathAdvanced(uint32_t aStringIdx)
{
	auto stringBounds = 50 + aStringIdx * 20;
	Path stringPath;
	stringPath.startNewSubPath(0, stringBounds);

	auto spacing = getWidth() / double(numPointsStrings[aStringIdx] - stringBoundaryX[aStringIdx]);
	auto x = spacing;

	for (int y = stringBoundaryX[aStringIdx]; y < numPointsStrings[aStringIdx]; y++)
	{
		float newY = uStrings[aStringIdx][y] *10 + stringBounds;
		if (isnan(newY))
			newY = 0;
		stringPath.lineTo(x, newY);
		x += spacing;

		//Connections
		int con = std::get<2>(connectionPointsIdx[aStringIdx]);
		if (y == con - 2)
		{
			std::get<2>(connectionPointsCoords[aStringIdx]) = getWidth() * (214.0 / 256.0);
			std::get<3>(connectionPointsCoords[aStringIdx]) = newY;
		}
	}
	stringPath.lineTo(getWidth(), stringBounds);

	return stringPath;
}
//==============================================================================
void MainComponent::paint (juce::Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));

	if (isInit && isPaint)
	{

		// You can add your drawing code here!
		g.setColour(Colours::mediumpurple);
		for (uint32_t i = 0; i != 13; ++i)
		{
			g.strokePath(generateStringPathAdvanced(i), PathStrokeType(4.0f));
		}

		int plateComponenetWidth = getWidth();
		int plateComponenetheight = getHeight() / 4;

		int Nx = 217; // 405 - 154 = 251
		int Ny = 121; //486 - 5 = 481
		float stateWidth = plateComponenetWidth / static_cast<double> (Nx - 4);
		float stateHeight = plateComponenetheight / static_cast<double> (Ny - 4);
		float tempStateHeight = plateComponenetheight / static_cast<double> (236 - 4);
		int scaling = 10000;

		for (int x = 0; x < Nx; ++x)
		{
			for (int y = 0; y < Ny; ++y)
			{
				int cVal = clamp(255 * 0.5 * (uPlate[(y*Nx) + x] * scaling + 1), 0, 255);
				g.setColour(Colour::fromRGBA(cVal, cVal, cVal, 127));
				g.fillRect((x - 2) * stateWidth, (y - 2) * stateHeight + (getHeight()/2), stateWidth, stateHeight);
				for (int c = 0; c < connectionPointsIdx.size(); ++c)
				{
					int conFirst = std::get<0>(connectionPointsIdx[c]);
					int conSecond = std::get<1>(connectionPointsIdx[c]);
					if (x == conFirst - 96 && y == conSecond - 5)
					{
						g.setColour(Colours::orange);
						//g.drawRect((x - 2) * stateWidth, (y - 2) * stateHeight, stateWidth*3, stateHeight*3);

						std::get<0>(connectionPointsCoords[c]) = (x - 2) * stateWidth;
						std::get<1>(connectionPointsCoords[c]) = (y - 2) * tempStateHeight + (getHeight() / 2);
					}
				}
				/*for (int c = 0; c < cpIdx.size(); ++c)
				{
					auto[cpX, cpY] = cpIdx[c];
					if (x == cpX && y == cpY)
					{
						g.setColour(Colours::orange);
						g.drawRect((x - 2) * stateWidth, (y - 2) * stateHeight, stateWidth, stateHeight);
					}
				}*/

			}
		}

		for (int c = 0; c < connectionPointsCoords.size(); ++c)
		{
			int conFirst = std::get<0>(connectionPointsCoords[c]);
			int conSecond = std::get<1>(connectionPointsCoords[c]);
			int conThird = std::get<2>(connectionPointsCoords[c]);
			int conFourth = std::get<3>(connectionPointsCoords[c]);

			float dashPattern[2];
			dashPattern[0] = 3.0;
			dashPattern[1] = 5.0;
			Line<float> connectionLine(conFirst, conSecond, conThird, conFourth);
			g.setColour(Colours::orange);
			g.drawDashedLine(connectionLine, dashPattern, 2, dashPattern[0], 0);
		}
	}
}

void MainComponent::resized()
{
	auto rect = getLocalBounds();

	audioSetupComp.setBounds(rect.removeFromLeft(proportionOfWidth(0.6f)));
	rect.reduce(10, 10);
	audioSetupComp.setVisible(false);

	auto topLine(rect.removeFromTop(20));
	cpuUsageLabel.setBounds(topLine.removeFromLeft(topLine.getWidth() / 2));
	cpuUsageText.setBounds(topLine);
	rect.removeFromTop(20);

	diagnosticsBox.setBounds(rect);
    // This is called when the MainContentComponent is resized.
    // If you add any child components, this is where you should
    // update their positions.
	uint32_t sliderLeft = 50;
	sldPropagationOne.setBounds(100, 750, getWidth() - 60, 40);
	sldDampingOne.setBounds(sldPropagationOne.getX(), sldPropagationOne.getY() + 40, sldPropagationOne.getWidth() - sliderLeft - 10, 40);
	sldDampingTwo.setBounds(sldDampingOne.getX(), sldDampingOne.getY() + 40, sldPropagationOne.getWidth() - sliderLeft - 10, 40);
	sldPropagationPlate.setBounds(sldDampingTwo.getX(), sldDampingTwo.getY() + 40, getWidth() - 60, 40);
	sldDampingPlateOne.setBounds(sldPropagationPlate.getX(), sldPropagationPlate.getY() + 40, sldPropagationPlate.getWidth() - sliderLeft - 10, 40);
	sldDampingPlateTwo.setBounds(sldDampingPlateOne.getX(), sldDampingPlateOne.getY() + 40, sldDampingPlateOne.getWidth() - sliderLeft - 10, 40);
	sldInputDuration.setBounds(sldDampingPlateTwo.getX(), sldDampingPlateTwo.getY() + 40, sldDampingPlateTwo.getWidth() - sliderLeft - 10, 40);
}

//Interface//
void MainComponent::buttonClicked(Button* btn)
{
	
}
void MainComponent::sliderValueChanged(Slider* sld)
{

}
void MainComponent::sliderDragEnded(Slider* sld)
{
	if (sld == &sldPropagationOne)
	{
		mutexInit.lock();

		_strKappa = sldPropagationOne.getValue();

		simulationModel->resetState();
		setDefaultStringCoeffs(_strKappa, _strSigmaZero, _strSigmaOne);

		//Update buffer of Coefficients.
		simulationModel->updateCoefficients(coeffs, numCoeffs_);

		mutexInit.unlock();
	}

	if (sld == &sldDampingOne)
	{
		mutexInit.lock();

		_strSigmaZero = sldDampingOne.getValue();

		simulationModel->resetState();
		setDefaultStringCoeffs(_strKappa, _strSigmaZero, _strSigmaOne);

		//Update buffer of Coefficients.
		simulationModel->updateCoefficients(coeffs, numCoeffs_);

		mutexInit.unlock();
	}
	if (sld == &sldPropagationTwo)
	{
		//mutexSensel.lock();

		//simulationModel->updateCoefficient("lambdaTwo", 11, sldPropagationTwo.getValue());

		//mutexSensel.unlock();
	}

	if (sld == &sldDampingTwo)
	{
		mutexInit.lock();

		_strSigmaOne = sldDampingTwo.getValue();

		simulationModel->resetState();
		setDefaultStringCoeffs(_strKappa, _strSigmaZero, _strSigmaOne);

		//Update buffer of Coefficients.
		simulationModel->updateCoefficients(coeffs, numCoeffs_);

		mutexInit.unlock();
	}
	if (sld == &sldPropagationPlate)
	{
		mutexInit.lock();

		_plateKappa = sldPropagationPlate.getValue();

		simulationModel->resetState();
		setPlateCoeffs(_plateKappa, _plateSigmaZero, _plateSigmaOne);

		//Update buffer of Coefficients.
		simulationModel->updateCoefficients(coeffs, numCoeffs_);

		mutexInit.unlock();
	}
	if (sld == &sldDampingPlateOne)
	{
		mutexInit.lock();

		_plateSigmaZero = sldDampingPlateOne.getValue();

		simulationModel->resetState();
		setPlateCoeffs(_plateKappa, _plateSigmaZero, _plateSigmaOne);

		//Update buffer of Coefficients.
		simulationModel->updateCoefficients(coeffs, numCoeffs_);

		mutexInit.unlock();
	}
	if (sld == &sldDampingPlateTwo)
	{
		mutexInit.lock();

		_plateSigmaOne = sldDampingPlateTwo.getValue();

		simulationModel->resetState();
		setPlateCoeffs(_plateKappa, _plateSigmaZero, _plateSigmaOne);

		//Update buffer of Coefficients.
		simulationModel->updateCoefficients(coeffs, numCoeffs_);

		mutexInit.unlock();
	}

	if (sld == &sldInputDuration)
	{
		//mutexSensel.lock();

		exciteDuration = sldInputDuration.getValue();
		wavetableExciter_.setDuration(exciteDuration);

		//mutexSensel.unlock();
	}
}

void MainComponent::hiResTimerCallback()
{
	if (isInit)
	{
		senselInterfaceOne->check();
		// String - 512
		//inputPos[0] = 7;
		//inputPos[1] = 9;
		//inputPos[0] = 8;
		//inputPos[1] = 6;
		// String - 64
		//inputPos[0] = 4;
		//inputPos[1] = 2;
		//inputPos[0] = 10;
		//inputPos[1] = 6;

		//Detect strings to play
		unsigned int fingerCount = senselInterfaceOne->contactAmount;
		for (int j = 0; j != fingerCount; ++j)
		{
			for (int i = 0; i != 13; ++i)
			{
				if (senselInterfaceOne->fingers[j].x > (1.0 / 13.0) * (i) && senselInterfaceOne->fingers[j].x < (1.0 / 13.0) * (i + 1))
				{
					currentString = i;
					inputPos[0] = inputCoords[i];
					inputPos[1] = stringBoundaryX[i] + (senselInterfaceOne->fingers[j].y * (232 - stringBoundaryX[i]));
					simulationModel->setInputPosition(inputPos);
				}
			}
			if (senselInterfaceOne->contactAmount > 0 && (senselInterfaceOne->fingers[j].state == CONTACT_START || lastString != currentString))
			{
				isPlayed_ = false;
				isExcite = true;
				stringExciter_.excite();
				wavetableExciter_.resetExcitation();
				lastString = currentString;
			}
		}

		senselInterfaceTwo->check();
		inputPos[0] = senselInterfaceTwo->fingers[0].x * 121 + 96;							//256 model.
		inputPos[1] = senselInterfaceTwo->fingers[0].y * 236 + 5;
		if (senselInterfaceTwo->contactAmount > 0 && (senselInterfaceTwo->fingers[0].state))
		{
			simulationModel->setInputPosition(inputPos);
			isPlayed_ = false;
			isExcite = true;
			stringExciter_.excite();
			wavetableExciter_.resetExcitation();
		}

		//auto cpu = deviceManager.getCpuUsage() * 100;
		//cpuUsageText.setText(juce::String(cpu, 6) + " %", juce::dontSendNotification);
	}
}