#pragma OPENCL EXTENSION cl_khr_fp64 : enable
int rem(int x, int y)
{
    return (x % y + y) % y;
}

__kernel
void fdtdKernel(__global int* idGrid, __global float* modelGrid, __global float* boundaryGrid, int idxRotate, int idxSample, __global float* input, __global float* output, int inputPosition, __global int* outputPosition, double lambdaFive, double lambdaTwo, double lambdaFour, double lambdaSix, double lambdaThree, double lambdaOne, float strLambdaOne, float strLambdaFive, float strLambdaTwo, float strLambdaThree, float strLambdaFour, float damp)
{
	//Rotation Index into model grid//
	int gridSize = get_global_size(0) * get_global_size(1);
    
	int rotation0 = gridSize * rem(idxRotate+0, 3);
	int rotationM1 = gridSize * rem(idxRotate+-1, 3);
	int rotation1 = gridSize * rem(idxRotate+1, 3);
	
    
	//Get index for current and neighbouring nodes//
	int t0x0y0Idx = rotation0 + ((get_global_id(1)+0) * get_global_size(0) + get_global_id(0)+0);
	int t0x1y0Idx = rotation0 + ((get_global_id(1)+1) * get_global_size(0) + get_global_id(0)+0);
	int t0xM1y0Idx = rotation0 + ((get_global_id(1)+-1) * get_global_size(0) + get_global_id(0)+0);
	int t0x0y1Idx = rotation0 + ((get_global_id(1)+0) * get_global_size(0) + get_global_id(0)+1);
	int t0x0yM1Idx = rotation0 + ((get_global_id(1)+0) * get_global_size(0) + get_global_id(0)+-1);
	int t0x1y1Idx = rotation0 + ((get_global_id(1)+1) * get_global_size(0) + get_global_id(0)+1);
	int t0xM1yM1Idx = rotation0 + ((get_global_id(1)+-1) * get_global_size(0) + get_global_id(0)+-1);
	int t0x1yM1Idx = rotation0 + ((get_global_id(1)+1) * get_global_size(0) + get_global_id(0)+-1);
	int t0xM1y1Idx = rotation0 + ((get_global_id(1)+-1) * get_global_size(0) + get_global_id(0)+1);
	int t0x2y0Idx = rotation0 + ((get_global_id(1)+2) * get_global_size(0) + get_global_id(0)+0);
	int t0xM2y0Idx = rotation0 + ((get_global_id(1)+-2) * get_global_size(0) + get_global_id(0)+0);
	int t0x0y2Idx = rotation0 + ((get_global_id(1)+0) * get_global_size(0) + get_global_id(0)+2);
	int t0x0yM2Idx = rotation0 + ((get_global_id(1)+0) * get_global_size(0) + get_global_id(0)+-2);
	int tM1x0y0Idx = rotationM1 + ((get_global_id(1)+0) * get_global_size(0) + get_global_id(0)+0);
	int tM1x1y0Idx = rotationM1 + ((get_global_id(1)+1) * get_global_size(0) + get_global_id(0)+0);
	int tM1xM1y0Idx = rotationM1 + ((get_global_id(1)+-1) * get_global_size(0) + get_global_id(0)+0);
	int tM1x0y1Idx = rotationM1 + ((get_global_id(1)+0) * get_global_size(0) + get_global_id(0)+1);
	int tM1x0yM1Idx = rotationM1 + ((get_global_id(1)+0) * get_global_size(0) + get_global_id(0)+-1);
	
	int t1x0y0Idx = rotation1 + ((get_global_id(1)) * get_global_size(0) + get_global_id(0));

	//Boundary condition evaluates neighbours in preperation for equation//
	//@ToDo - Make new timestep value autogenerated?//
	float t1x0y0;
	float t0x0y0;
	float t0x1y0;
	float t0xM1y0;
	float t0x0y1;
	float t0x0yM1;
	float t0x1y1;
	float t0xM1yM1;
	float t0x1yM1;
	float t0xM1y1;
	float t0x2y0;
	float t0xM2y0;
	float t0x0y2;
	float t0x0yM2;
	float tM1x0y0;
	float tM1x1y0;
	float tM1xM1y0;
	float tM1x0y1;
	float tM1x0yM1;
	
	tM1x0y0 = modelGrid[tM1x0y0Idx];
	t0x0y0 = modelGrid[t0x0y0Idx];
	

	int centreIdx = (get_global_id(1)) * get_global_size(0) + get_global_id(0);
    int leftIdx = (get_global_id(1)-1) * get_global_size(0) + get_global_id(0);
    int rightIdx = (get_global_id(1)-1) * get_global_size(0) + get_global_id(0);
    int upIdx = (get_global_id(1)) * get_global_size(0) + get_global_id(0)-1;
    int downIdx = (get_global_id(1)) * get_global_size(0) + get_global_id(0)+1;

    int leftLeftIdx = (get_global_id(1)-2) * get_global_size(0) + get_global_id(0);
    int rightRightIdx = (get_global_id(1)+2) * get_global_size(0) + get_global_id(0);
    int upUpIdx = (get_global_id(1)) * get_global_size(0) + get_global_id(0)-2;
    int downDownIdx = (get_global_id(1)) * get_global_size(0) + get_global_id(0)+2;

    int leftUpIdx = (get_global_id(1)-1) * get_global_size(0) + get_global_id(0)-1;
    int rightUpIdx = (get_global_id(1)+1) * get_global_size(0) + get_global_id(0)-1;
    int leftDownIdx = (get_global_id(1)-1) * get_global_size(0) + get_global_id(0)+1;
    int rightDownIdx = (get_global_id(1)+1) * get_global_size(0) + get_global_id(0)+1;

    // Hybird Boundary?
    // t0x0yM1 = modelGrid[t0x0yM1Idx] * (1-boundaryGrid[upIdx]) +  modelGrid[centreIdx] * (boundaryGrid[upIdx]);
    // t0x1y0 = modelGrid[t0x1y0Idx] * (1-boundaryGrid[rightIdx]) +  modelGrid[centreIdx] * (boundaryGrid[rightIdx]);
    // t0x0y1 = modelGrid[t0x0y1Idx]* (1-boundaryGrid[downIdx]) +  modelGrid[centreIdx] * (boundaryGrid[downIdx]);
    // t0xM1y0 = modelGrid[t0xM1y0Idx]* (1-boundaryGrid[leftIdx])+  modelGrid[centreIdx] * (boundaryGrid[leftIdx]);

    // t0x1y1 = modelGrid[t0x1y1Idx]* (1-boundaryGrid[rightDownIdx])+  modelGrid[centreIdx] * (boundaryGrid[rightDownIdx]);
    // t0x1yM1 = modelGrid[t0x1yM1Idx]* (1-boundaryGrid[rightUpIdx])+  modelGrid[centreIdx] * (boundaryGrid[rightUpIdx]);
    // t0xM1y1 = modelGrid[t0xM1y1Idx]* (1-boundaryGrid[leftDownIdx])+  modelGrid[centreIdx] * (boundaryGrid[leftDownIdx]);
    // t0xM1yM1 = modelGrid[t0xM1yM1Idx]* (1-boundaryGrid[leftUpIdx])+  modelGrid[centreIdx] * (boundaryGrid[leftUpIdx]);

    // t0x0y2 = modelGrid[t0x0y2Idx]* (1-boundaryGrid[downDownIdx])+  modelGrid[centreIdx] * (boundaryGrid[downDownIdx]);
    // t0x2y0 = modelGrid[t0x2y0Idx]* (1-boundaryGrid[rightRightIdx])+  modelGrid[centreIdx] * (boundaryGrid[rightRightIdx]);
    // t0xM2y0 = modelGrid[t0xM2y0Idx]* (1-boundaryGrid[leftLeftIdx])+  modelGrid[centreIdx] * (boundaryGrid[leftLeftIdx]);
    // t0x0yM2 = modelGrid[t0x0yM2Idx]* (1-boundaryGrid[upUpIdx])+  modelGrid[centreIdx] * (boundaryGrid[upUpIdx]);

    // tM1x1y0 = modelGrid[tM1x1y0Idx]* (1-boundaryGrid[rightIdx])+  modelGrid[centreIdx] * (boundaryGrid[downIdx]);
    // tM1xM1y0 = modelGrid[tM1xM1y0Idx]* (1-boundaryGrid[leftIdx])+  modelGrid[centreIdx] * (boundaryGrid[downIdx]);
    // tM1x0y1 = modelGrid[tM1x0y1Idx]* (1-boundaryGrid[downIdx])+  modelGrid[centreIdx] * (boundaryGrid[downIdx]);
    // tM1x0yM1 = modelGrid[tM1x0yM1Idx]* (1-boundaryGrid[upIdx])+  modelGrid[centreIdx] * (boundaryGrid[downIdx]);

    // Dirichlet Boundary?
    // t0x0yM1 = modelGrid[t0x0yM1Idx] * (1-boundaryGrid[upIdx]);
    // t0x1y0 = modelGrid[t0x1y0Idx] * (1-boundaryGrid[rightIdx]);
    // t0x0y1 = modelGrid[t0x0y1Idx]* (1-boundaryGrid[downIdx]);
    // t0xM1y0 = modelGrid[t0xM1y0Idx]* (1-boundaryGrid[leftIdx]);

    // t0x1y1 = modelGrid[t0x1y1Idx]* (1-boundaryGrid[rightDownIdx]);
    // t0x1yM1 = modelGrid[t0x1yM1Idx]* (1-boundaryGrid[rightUpIdx]);
    // t0xM1y1 = modelGrid[t0xM1y1Idx]* (1-boundaryGrid[leftDownIdx]);
    // t0xM1yM1 = modelGrid[t0xM1yM1Idx]* (1-boundaryGrid[leftUpIdx]);

    // t0x0y2 = modelGrid[t0x0y2Idx]* (1-boundaryGrid[downDownIdx]);
    // t0x2y0 = modelGrid[t0x2y0Idx]* (1-boundaryGrid[rightRightIdx]);
    // t0xM2y0 = modelGrid[t0xM2y0Idx]* (1-boundaryGrid[leftLeftIdx]);
    // t0x0yM2 = modelGrid[t0x0yM2Idx]* (1-boundaryGrid[upUpIdx]);

    // tM1x1y0 = modelGrid[tM1x1y0Idx]* (1-boundaryGrid[rightIdx]);
    // tM1xM1y0 = modelGrid[tM1xM1y0Idx]* (1-boundaryGrid[leftIdx]);
    // tM1x0y1 = modelGrid[tM1x0y1Idx]* (1-boundaryGrid[downIdx]);
    // tM1x0yM1 = modelGrid[tM1x0yM1Idx]* (1-boundaryGrid[upIdx]);

    // No boundary
    t0xM2y0 = modelGrid[t0xM2y0Idx];
    t0x0y2 = modelGrid[t0x0y2Idx];
    t0x0yM2 = modelGrid[t0x0yM2Idx];
    tM1x1y0 = modelGrid[tM1x1y0Idx];
    tM1xM1y0 = modelGrid[tM1xM1y0Idx];
    tM1x0y1 = modelGrid[tM1x0y1Idx];
    tM1x0yM1 = modelGrid[tM1x0yM1Idx];
    t0x0y1 = modelGrid[t0x0y1Idx];
    t0x1y0 = modelGrid[t0x1y0Idx];
    t0xM1y0 = modelGrid[t0xM1y0Idx];
    t0x0yM1 = modelGrid[t0x0yM1Idx];
    t0x1y1 = modelGrid[t0x1y1Idx];
    t0xM1yM1 = modelGrid[t0xM1yM1Idx];
    t0x1yM1 = modelGrid[t0x1yM1Idx];
    t0xM1y1 = modelGrid[t0xM1y1Idx];
    t0x2y0 = modelGrid[t0x2y0Idx];
	
	//Calculate the next pressure value//
	if(idGrid[centreIdx] == 0)
{t1x0y0 = 0.0;}
if(idGrid[centreIdx] == 1) {
		t1x0y0 = (((lambdaOne*t0x0y0)+(lambdaTwo*(t0x1y0+t0xM1y0+t0x0y1+t0x0yM1))+(lambdaThree*(t0x1y1+t0xM1yM1+t0x1yM1+t0xM1y1))+(lambdaFour*(t0x2y0+t0xM2y0+t0x0y2+t0x0yM2))+(lambdaFive*tM1x0y0)+(lambdaSix*(t0x1y0+t0xM1y0+t0x0y1+t0x0yM1-tM1x1y0-tM1xM1y0-tM1x0y1-tM1x0yM1)))) * damp;
}
	if(idGrid[centreIdx] == 2) {
		t1x0y0 = ((strLambdaOne*modelGrid[t0x0y0Idx])+(strLambdaTwo*(modelGrid[t0x1y0Idx]+modelGrid[t0xM1y0Idx]))-(strLambdaThree*(modelGrid[t0x2y0Idx]+modelGrid[t0xM2y0Idx]))+(strLambdaFour*modelGrid[tM1x0y0Idx])-(strLambdaFive*(modelGrid[tM1x1y0Idx]+modelGrid[tM1xM1y0Idx])));
}
if(idGrid[centreIdx] == 3) {
		t1x0y0 = ((strLambdaOne*modelGrid[t0x0y0Idx])+(strLambdaTwo*(modelGrid[t0x1y0Idx]+modelGrid[t0xM1y0Idx]))-(strLambdaThree*(modelGrid[t0x2y0Idx]+modelGrid[t0xM2y0Idx]))+(strLambdaFour*modelGrid[tM1x0y0Idx])-(strLambdaFive*(modelGrid[tM1x1y0Idx]+modelGrid[tM1xM1y0Idx])));
}
	
	//If the cell is the listener position, sets the next sound sample in buffer to value contained here//
	if(outputPosition[centreIdx] == 1)
	{
		output[idxSample] += t0x0y0;    //@ToDo - Make current timestep centre point auto generated?
	}
	
	if(centreIdx == inputPosition)	//If the position is an excitation...
	{
		t1x0y0 += input[idxSample];	//Input excitation value into point. Then increment to next excitation in next iteration.
	}
	
	modelGrid[t1x0y0Idx] = t1x0y0;
}

__kernel
void connectionsKernel(__global int* idGrid, __global float* modelGrid, __global float* boundaryGrid, int idxRotate)
{
    //Rotation Index into model grid//
	int gridSize = get_global_size(0) * get_global_size(1);
    
	int rotation0 = gridSize * rem(idxRotate+0, 3);
	int rotationM1 = gridSize * rem(idxRotate+-1, 3);
	int rotation1 = gridSize * rem(idxRotate+1, 3);

    int centreIdx = (get_global_id(1)) * get_global_size(0) + get_global_id(0);
    int t0x0y0Idx = rotation0 + ((get_global_id(1)+0) * get_global_size(0) + get_global_id(0)+0);
    int t1x0y0Idx = rotation1 + ((get_global_id(1)) * get_global_size(0) + get_global_id(0));
    if(centreIdx == 1056)
    {
        float intermediateOne = (modelGrid[t1x0y0Idx] - modelGrid[rotation1 + 3204]) / 1000.0;
        float intermediateTwo = (modelGrid[t1x0y0Idx] - modelGrid[rotation1 + 3204]) / 1000.0;

        
        modelGrid[t1x0y0Idx] += intermediateOne;
        modelGrid[rotation1 + 3204] -= intermediateTwo;
    }

    if(centreIdx == 1560)
    {
        float intermediateOne = (modelGrid[t1x0y0Idx] - modelGrid[rotation1 + 330]) / 1200.0;
        float intermediateTwo = (modelGrid[t1x0y0Idx] - modelGrid[rotation1 + 330]) / 1200.0;

        
        modelGrid[t1x0y0Idx] += intermediateOne;
        modelGrid[rotation1 + 330] -= intermediateTwo;
    }
}