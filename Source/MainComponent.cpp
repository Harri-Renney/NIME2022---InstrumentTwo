#include "MainComponent.h"

float secondInput[44100];
double xposInput;
double yposInput;
int inputPos[2]{ 0, 0 };
int outputPos[2]{ 0, 0 };

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
	addAndMakeVisible(diagnosticsBox);

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
	sldPropagationOne.setRange(0.0, 25000, 1.0);
	sldPropagationOne.setValue(50);
	sldPropagationOne.addListener(this);
	addAndMakeVisible(sldDampingOne);
	sldDampingOne.setRange(0.0, 0.5, 0.000001);
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

	addAndMakeVisible(sldInputDuration);
	sldInputDuration.setRange(0.0, 1000, 1.0);
	sldInputDuration.setValue(10);
	sldInputDuration.addListener(this);

	//Labels//
	addAndMakeVisible(lblPropagationOne);
	lblPropagationOne.setText("Propagation", dontSendNotification);
	lblPropagationOne.attachToComponent(&sldPropagationOne, true);
	addAndMakeVisible(lblDampingOne);
	lblDampingOne.setText("Damping", dontSendNotification);
	lblDampingOne.attachToComponent(&sldDampingOne, true);
	addAndMakeVisible(lblPropagationTwo);
	lblPropagationTwo.setText("Propagation Two", dontSendNotification);
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
	double stringFundamentalFrequency = 210;
	double stringGenDamp = 1.000;
	double stringFreqDamp = 0.005;
	// 64
	//double stringFundamentalFrequency = 210;
	//double stringGenDamp = 1.000;
	//double stringFreqDamp = 0.005;
	setStringCoeffs(stringFundamentalFrequency, stringGenDamp, stringFreqDamp);

	//512
	double plateFundamentalFrequency = 1000;
	double plateGenDamp = 0.1;
	double plateFreqDamp = 0.07;
	//64
	//double plateFundamentalFrequency = 1;
	//double plateGenDamp = 0.1;
	//double plateFreqDamp = 0.005;
	setPlateCoeffs(plateFundamentalFrequency, plateGenDamp, plateFreqDamp);

	//Setup output position
	// String - 512
	outputPos[0] = 7;
	outputPos[1] = 464;
	// String - 64
	//outputPos[0] = 4;
	//outputPos[1] = 51;
	//outputPos[0] = 10;
	//outputPos[1] = 40;
	outputPos[0] = simulationModel->getModelHeight() / 2.0;
	outputPos[1] = simulationModel->getModelWidth() / 2.0;
	simulationModel->setOutputPosition(outputPos);
	//outputPos[0] = simulationModel->getModelHeight() / 2.0;
	//outputPos[1] = (simulationModel->getModelWidth() / 4.0) * 2.5;
	//simulationModel->setOutputPosition(outputPos);

	//Timer callback
	startTimer(1);
	isInit = true;
	mutexInit.unlock();
}

MainComponent::~MainComponent()
{
    // This shuts down the audio device and clears the audio source.
    shutdownAudio();
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
	double gamma = aFundFreq * 2;				// Wave speed
	double sampleRate = 44100;
	double k = 1.0f / sampleRate;				//Timestep.

	double s0 = aSigmaOne;						// Frequency-independent damping
	double s1 = aSigmaTwo;						// Frequency-dependent damping

	// Courant numbers
	double N = 28;								//478 // @TODO - Needs to be the number of grid points making up string. Grid is 512, but string is less. How to calculate this??
	double kappa = 2.0;
	double h = 1.0 / N;	
	double lambdaSq = pow(gamma * k / h, 2);
	double muSq = pow(k * kappa / (h * h), 2);

	double B1 = s0 * k;
	double B2 = (2 * s1 * k) / (h * h);

	double b1 = 2.0 / (k * k);
	double b2 = (2 * s1) / (k * h * h);

	double D = 1.0 / (1.0 + s0 * k);

	float A1 = 2 - 2 * lambdaSq - 6 * muSq - 2 * B2;
	float A2 = lambdaSq + 4 * muSq + B2;
	float A3 = muSq;
	float A4 = B1 - 1 + 2 * B2;
	float A5 = B2;

	A1 *= D;
	A2 *= D;
	A3 *= D;
	A4 *= D;
	A5 *= D;

	simulationModel->updateCoefficient("strLambdaOne", 15, A1);
	simulationModel->updateCoefficient("strLambdaFive", 16, A5);
	simulationModel->updateCoefficient("strLambdaTwo", 17, A2);
	simulationModel->updateCoefficient("strLambdaThree", 18, A3);
	simulationModel->updateCoefficient("strLambdaFour", 19, A4);
}
void MainComponent::setPlateCoeffs(double aKappa, double aSigmaOne, double aSigmaTwo)
{
	_kappa = aKappa;
	_sigmaZero = aSigmaOne;
	_sigmaOne = aSigmaTwo;

	//double maxKappaSq = 25000;	//512
	double maxKappaSq = 25000 * 25000;		//64
	double kappaSq = clamp(_kappa*_kappa, 0.1, maxKappaSq);
	double sampleRate = 44100;
	double k = 1.0f / sampleRate;			//Timestep.

	double h = 2 * sqrt(k*(_sigmaOne * _sigmaOne + sqrt(kappaSq + _sigmaOne * _sigmaOne)));
	//double h = 2 * sqrt(k*(aSigmaTwo * aSigmaTwo + sqrt(maxKappaSq + aSigmaTwo * aSigmaTwo)));
	//double h = 0.0095;
	//double h = 0.30117005389;

	double d = 1.0f / (1.0f + _sigmaZero * k);
	float lambdaFour = -(kappaSq * k * k) / (h * h * h * h) * d;		// (mu^2)
	float lambdaThree = lambdaFour * 2.0f;
	float lambdaTwo = lambdaFour * -8.0f;								// (8 * mu^2)
	float S = (2.0f * _sigmaOne * k) / (h * h);							// S = 2 * sigmaOne * kappa / h^2
	float lambdaOne = (2.0f - 4.0f * S + 20.0f * lambdaFour) * d;
	float lambdaFive = (_sigmaZero * k - 1.0f + 4.0f * S) * d;			//(sigmaZero * k - 1 + 4 * S)
	float lambdaSix = S * d;
	float C4 = (k * k) * d;

	simulationModel->updateCoefficient("lambdaFive", 9, lambdaFive);
	simulationModel->updateCoefficient("lambdaTwo", 10, lambdaTwo);
	simulationModel->updateCoefficient("lambdaFour", 11, lambdaFour);
	simulationModel->updateCoefficient("lambdaSix", 12, lambdaSix);
	simulationModel->updateCoefficient("lambdaThree", 13, lambdaThree);
	simulationModel->updateCoefficient("lambdaOne", 14, lambdaOne);
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
		for (int j = 0; j != bufferToFill.numSamples; ++j)
		{
			//inputExcitation[j] = sineExciter_.getNextSample();
			if (wavetableExciter_.isExcitation())
				inputExcitation[j] = wavetableExciter_.getNextSample() * 0.1;
			else
				inputExcitation[j] = 0;
		}

		// For more details, see the help for AudioProcessor::getNextAudioBlock()
		simulationModel->fillBuffer(inputExcitation, leftBuffer, bufferToFill.numSamples);
		memcpy(rightBuffer, leftBuffer, bufferToFill.numSamples * sizeof(float));

		counter += (bufferToFill.numSamples);
		if (counter > framerate)
		{
			simulationModel->renderSimulation();
			counter = 0;
		}

		mutexInit.unlock();
	}
}

void MainComponent::releaseResources()
{
    // This will be called when the audio device stops, or when it is being
    // restarted due to a setting change.

    // For more details, see the help for AudioProcessor::releaseResources()
}

//==============================================================================
void MainComponent::paint (juce::Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));

    // You can add your drawing code here!
}

void MainComponent::resized()
{
	auto rect = getLocalBounds();

	audioSetupComp.setBounds(rect.removeFromLeft(proportionOfWidth(0.6f)));
	rect.reduce(10, 10);

	auto topLine(rect.removeFromTop(20));
	cpuUsageLabel.setBounds(topLine.removeFromLeft(topLine.getWidth() / 2));
	cpuUsageText.setBounds(topLine);
	rect.removeFromTop(20);

	diagnosticsBox.setBounds(rect);
    // This is called when the MainContentComponent is resized.
    // If you add any child components, this is where you should
    // update their positions.
	uint32_t sliderLeft = 50;
	btnCreateDrum.setBounds(30, 350, getWidth() - 60, 40);
	sldGridWidth.setBounds(btnCreateDrum.getX() + sliderLeft, btnCreateDrum.getY() + 40, btnCreateDrum.getWidth() - sliderLeft - 10, 40);
	sldGridHeight.setBounds(sldGridWidth.getX(), sldGridWidth.getY() + 40, btnCreateDrum.getWidth() - sliderLeft - 10, 40);
	sldPropagationOne.setBounds(sldGridHeight.getX(), sldGridHeight.getY() + 40, btnCreateDrum.getWidth() - sliderLeft - 10, 40);
	sldDampingOne.setBounds(sldPropagationOne.getX(), sldPropagationOne.getY() + 40, btnCreateDrum.getWidth() - sliderLeft - 10, 40);
	sldPropagationTwo.setBounds(sldDampingOne.getX(), sldDampingOne.getY() + 40, btnCreateDrum.getWidth() - sliderLeft - 10, 40);
	sldDampingTwo.setBounds(sldPropagationTwo.getX(), sldPropagationTwo.getY() + 40, btnCreateDrum.getWidth() - sliderLeft - 10, 40);
	sldInputDuration.setBounds(sldDampingTwo.getX(), sldDampingTwo.getY() + 40, btnCreateDrum.getWidth() - sliderLeft - 10, 40);
	btnExcite.setBounds(btnCreateDrum.getX(), sldInputDuration.getY() + 40, btnCreateDrum.getWidth(), 40);
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

		_kappa = sldPropagationOne.getValue();

		setPlateCoeffs(_kappa, _sigmaZero, _sigmaOne);

		mutexInit.unlock();
	}

	if (sld == &sldDampingOne)
	{
		mutexInit.lock();

		_sigmaZero = sldDampingOne.getValue();

		setPlateCoeffs(_kappa, _sigmaZero, _sigmaOne);

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

		_sigmaOne = sldDampingTwo.getValue();

		setPlateCoeffs(_kappa, _sigmaZero, _sigmaOne);

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
	senselInterface.check();
	inputPos[0] = senselInterface.fingers[0].x * simulationModel->getModelWidth();
	inputPos[1] = senselInterface.fingers[0].y * simulationModel->getModelHeight();
	// String - 512
	//inputPos[0] = 7;
	//inputPos[1] = 9;
	// String - 64
	//inputPos[0] = 4;
	//inputPos[1] = 2;
	//inputPos[0] = 10;
	//inputPos[1] = 6;

	//if (senselInterface.fingers[0].x < 0.5)
	//{
	//	inputPos[0] = 4;
	//	inputPos[1] = 2;
	//}
	//else
	//{
	//	inputPos[0] = 10;
	//	inputPos[1] = 6;
	//}

	simulationModel->setInputPosition(inputPos);
	if (senselInterface.contactAmount > 0 && senselInterface.fingers[0].state == CONTACT_START)
	{
		wavetableExciter_.resetExcitation();
	}

	//auto cpu = deviceManager.getCpuUsage() * 100;
	//cpuUsageText.setText(juce::String(cpu, 6) + " %", juce::dontSendNotification);
}