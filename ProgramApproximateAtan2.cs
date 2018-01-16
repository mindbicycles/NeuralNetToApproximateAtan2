using System;
using System.Windows; 


//experiment with the notion of neural networks as universal function approximators by trying to approximate the output of the atan2 function using a neural net that takes as an input a normalized vector2 and outputs the angle in radians
//based on source Code From Neural Networks Using C# Succinctly by James McCaffrey, forked from https://github.com/mindbicycles/NeuralNetworksUsingCSharpSuccinctly/blob/master/Chapter4-BackProp/Program.cs 
class ProgramApproximateAtan2 {
    static void Main(string[] args) {

        Console.WriteLine("Creating a 2-4-8-1 neural network\n"); 
        int numInput = 2;
        int numHidden = 4;
        int numHidden2 = 8;
        int numOutput = 1;

        NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numHidden2, numOutput);

        int totalWeigthsAndBiases = (numInput * numHidden) + numHidden + 
                                    (numHidden * numHidden2) + numHidden2 +
                                    (numHidden2 * numOutput) + numOutput;

        Console.WriteLine("\ntotalWeigthsAndBiases = "+totalWeigthsAndBiases);

        Console.WriteLine("\nSetting fixed inputs = ");

        Random rnd1 = new Random();
        double[] weights = new double[totalWeigthsAndBiases];
        for (int i = 0; i < totalWeigthsAndBiases; i++)
        {
            weights[i] = rnd1.NextDouble();
        }

        Console.WriteLine("Setting dummy initial weights to:"); 
        ShowVector(weights, 8, 2, true); 
        nn.SetWeights(weights);
        double[] xValues = new double[2] { 1.0, 2.0}; 
        double[] tValues = new double[1] { 3.0 }; // dummy target outputs.

        Console.WriteLine("\nSetting fixed inputs = "); 
        ShowVector(xValues, 3, 1, true); 
        Console.WriteLine("Setting fixed target outputs = "); 
        ShowVector(tValues, 2, 4, true);

        double learnRate =  0.005; //0.05;
        double momentum = 0.01;
        int maxEpochs = 1000000;
        Console.WriteLine("\nSetting learning rate = " + learnRate.ToString("F2")); 
        Console.WriteLine("Setting momentum = " + momentum.ToString("F2")); 
        Console.WriteLine("Setting max epochs = " + maxEpochs + "\n");
        nn.FindWeights(tValues, xValues, learnRate, momentum, maxEpochs);
        double[] bestWeights = nn.GetWeights(); Console.WriteLine("\nBest weights found:"); 
        ShowVector(bestWeights, 8, 4, true);
        Console.WriteLine("\nEnd back-propagation demo\n");
        Console.ReadLine(); 
    } // Main

    public static void ShowVector(double[] vector, int valsPerRow, int decimals, bool newLine)
    {
        for (int i = 0; i < vector.Length; ++i) {
            if (i > 0 && i % valsPerRow == 0) Console.WriteLine("");
            Console.Write(vector[i].ToString("F" + decimals).PadLeft(decimals + 4) + " "); 
        }
        if (newLine == true) Console.WriteLine(""); 
    }

    public static void ShowMatrix(double[][] matrix, int decimals) 
    {
        int cols = matrix[0].Length;
        for (int i = 0; i < matrix.Length; ++i) // Each row.
            ShowVector(matrix[i], cols, decimals, true);
    }

} 

// Program class
public class NeuralNetwork
{
    private int numInput; 
    private int numHidden; 
    private int numHidden2; 
    private int numOutput;
    private double[] inputs;
    private double[][] ihWeights; 
    private double[] hBiases; 
    private double[] hOutputs;
    private double[][] hoWeights; 
    private double[] h2Biases; 
    private double[] h2Outputs;
    private double[][] h2oWeights; 
    private double[] oBiases; 
    private double[] outputs;
    private double[] oGrads; // Output gradients for back-propagation.
    private double[] hGrads; // Hidden gradients for back-propagation.
    private double[] h2Grads; // Hidden gradients for back-propagation.
    private double[][] ihPrevWeightsDelta; // For momentum. 
    private double[] hPrevBiasesDelta;
    private double[] h2PrevBiasesDelta;
    private double[][] hoPrevWeightsDelta;
    private double[][] h2oPrevWeightsDelta;
    private double[] oPrevBiasesDelta;

    public NeuralNetwork(int numInput, int numHidden, int numHidden2, int numOutput) {
        this.numInput = numInput; 
        this.numHidden = numHidden; 
        this.numHidden2 = numHidden2; 
        this.numOutput = numOutput;
        this.inputs = new double[numInput]; 
        this.ihWeights = MakeMatrix(numInput, numHidden); 

        this.hBiases = new double[numHidden]; 
        this.hOutputs = new double[numHidden];
        this.hoWeights = MakeMatrix(numHidden, numHidden2); 

        this.h2Biases = new double[numHidden2]; 
        this.h2Outputs = new double[numHidden2];
        this.h2oWeights = MakeMatrix(numHidden2, numOutput); 

        this.oBiases = new double[numOutput]; 
        this.outputs = new double[numOutput];
        oGrads = new double[numOutput]; 
        hGrads = new double[numHidden];
        h2Grads = new double[numHidden2];
        ihPrevWeightsDelta = MakeMatrix(numInput, numHidden); 
        hPrevBiasesDelta = new double[numHidden]; 
        h2PrevBiasesDelta = new double[numHidden2]; 
        hoPrevWeightsDelta = MakeMatrix(numHidden, numHidden2); 
        h2oPrevWeightsDelta = MakeMatrix(numHidden2, numOutput); 
        oPrevBiasesDelta = new double[numOutput];
        InitMatrix(ihPrevWeightsDelta, 0.011); 
        InitVector(hPrevBiasesDelta, 0.011); 
        InitMatrix(hoPrevWeightsDelta, 0.011); 
        InitVector(oPrevBiasesDelta, 0.011);
    }

    private static double[][] MakeMatrix(int rows, int cols) 
    {
        double[][] result = new double[rows][]; 
        for (int i = 0; i < rows; ++i)
            result[i] = new double[cols]; 

        return result;
    }

    private static void InitMatrix(double[][] matrix, double value) {
        int rows = matrix.Length;
        int cols = matrix[0].Length; 
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j) 
                matrix[i][j] = value;
    }

    private static void InitVector(double[] vector, double value) 
    {
        for (int i = 0; i < vector.Length; ++i) 
            vector[i] = value;
    }

    public void SetWeights(double[] weights) 
    {
        //int numWeights = (numInput * numHidden) + numHidden + // (numHidden * numOutput) + numOutput;
        //if (weights.Length != numWeights)
        // throw new Exception("Bad weights array");
        int k = 0; // Pointer into weights.
        for (int i = 0; i < numInput; ++i) 
            for (int j = 0; j < numHidden; ++j)
                ihWeights[i][j] = weights[k++];
        
        for (int i = 0; i < numHidden; ++i) 
            hBiases[i] = weights[k++];



        for (int i = 0; i < numHidden; ++i) 
            for (int j = 0; j < numHidden2; ++j)
                hoWeights[i][j] = weights[k++];

        for (int i = 0; i < numHidden2; ++i) 
            h2Biases[i] = weights[k++];




        for (int i = 0; i < numHidden2; ++i) 
            for (int j = 0; j < numOutput; ++j)
                h2oWeights[i][j] = weights[k++];
        
        for (int i = 0; i < numOutput; ++i) 
            oBiases[i] = weights[k++];
    }

    public double[] GetWeights() 
    {
        int numWeights = (numInput * numHidden) + numHidden +
                        (numHidden * numHidden2) + numHidden2 +
                        (numHidden2 * numOutput) + numOutput;
        double[] result = new double[numWeights];
        int k = 0;
        for (int i = 0; i < numInput; ++i)
            for (int j = 0; j < numHidden; ++j) 
                result[k++] = ihWeights[i][j];
        
        for (int i = 0; i < numHidden; ++i) 
            result[k++] = hBiases[i];


        for (int i = 0; i < numHidden; ++i)
            for (int j = 0; j < numHidden2; ++j) 
                result[k++] = hoWeights[i][j];

        for (int i = 0; i < numHidden2; ++i) 
            result[k++] = h2Biases[i];



        for (int i = 0; i < numHidden2; ++i) 
            for (int j = 0; j < numOutput; ++j)
                result[k++] = h2oWeights[i][j];
        
        for (int i = 0; i < numOutput; ++i) 
            result[k++] = oBiases[i];
        return result; 
    }

    private double[] ComputeOutputs(double[] xValues)
    {
        if (xValues.Length != numInput)
            throw new Exception("Bad xValues array");
        
        double[] hSums = new double[numHidden]; 
        double[] h2Sums = new double[numHidden2]; 
        double[] oSums = new double[numOutput];

        //copy input
        for (int i = 0; i < xValues.Length; ++i) 
            inputs[i] = xValues[i];

        //add weights and sum
        for (int j = 0; j < numHidden; ++j) 
            for (int i = 0; i < numInput; ++i)
            hSums[j] += inputs[i] * ihWeights[i][j];

        //add bias
        for (int i = 0; i < numHidden; ++i) 
            hSums[i] += hBiases[i];

        //activate hidden layer
        for (int i = 0; i < numHidden; ++i) 
            //hOutputs[i] = Logistic(hSums[i]);
            hOutputs[i] = ReLU(hSums[i]);
            //hOutputs[i] = HyperTan(hSums[i]);



        //add weights and sum
        for (int j = 0; j < numHidden2; ++j) 
            for (int i = 0; i < numHidden; ++i)
                h2Sums[j] += hOutputs[i] * hoWeights[i][j];

        //add bias
        for (int i = 0; i < numHidden2; ++i) 
            h2Sums[i] += h2Biases[i];

        //activate hidden layer
        for (int i = 0; i < numHidden2; ++i) 
            //h2Outputs[i] = Logistic(h2Sums[i]);
            h2Outputs[i] = ReLU(h2Sums[i]);
        //hOutputs[i] = HyperTan(hSums[i]);




        //add  weights and sum
        for (int j = 0; j < numOutput; ++j) 
            for (int i = 0; i < numHidden2; ++i)
            oSums[j] += h2Outputs[i] * h2oWeights[i][j];

        //add bias
        for (int i = 0; i < numOutput; ++i) 
            oSums[i] += oBiases[i];

        //double[] softOut = Softmax(oSums); // All outputs at once. 
        //double[] softOut = HyperTan(oSums); // All outputs at once. 

        //activate output layer
        for (int i = 0; i < outputs.Length; ++i)
            //outputs[i] = HyperTan(oSums[i]);
            //outputs[i] = Logistic(oSums[i]);
            outputs[i] = ReLU(oSums[i]);

        //copy results
        double[] result = new double[numOutput]; 
        for (int i = 0; i < outputs.Length; ++i)
            result[i] = outputs[i];

        return result; 
    }

    private static double HyperTan(double v) {
        if (v < -20.0) 
            return -1.0;
        else if (v > 20.0) 
            return 1.0;
        else 
            return Math.Tanh(v); 
    }

    private static double LeakyRelu(double x){
        if (x < 0.0) 
            return 0.01 * x; 
        else 
            return x;  
    }

    private static double LogSigmoid(double x) 
    {
        if (x < -45.0) return 0.0;
        else if (x > 45.0) return 1.0;
        else
            return 1.0 / (1.0 + Math.Exp(-x));
    }

    public double Logistic(double x)
    {
        return 1/(1+Math.Pow(Math.E,-x));
    }
    public double DLogistic(double x)
    {
        return Logistic(x)*(1-Logistic(x));
    }

    //Rectified Linear Unit from https://stackoverflow.com/questions/36384249/list-of-activation-functions-in-c-sharp
    private static double ReLU(double x)
    {
        return Math.Max(0,x);// x < 0 ? 0 : x;
    }


    private static double[] Softmax(double[] oSums) {
        double max = oSums[0];
        for (int i = 0; i < oSums.Length; ++i)
            if (oSums[i] > max) max = oSums[i];
        double scale = 0.0;
        for (int i = 0; i < oSums.Length; ++i)
            scale += Math.Exp(oSums[i] - max);
        double[] result = new double[oSums.Length]; 
        for (int i = 0; i < oSums.Length; ++i)
            result[i] = Math.Exp(oSums[i] - max) / scale; 
        return result; // xi sum to 1.0.
    }

    public void FindWeights(double[] tValues, double[] xValues, double learnRate, double momentum, int maxEpochs)
    {
        // Call UpdateWeights maxEpoch times.
        int epoch = 0;
        Random rnd1 = new Random();
        double loss = 0.0;
        double prevLoss = -1.0;
        while (epoch <= maxEpochs)
        {
            
            //override values
            //xValues[0] = rnd1.NextDouble();
            //xValues[1] = rnd1.NextDouble();
            Vector vectorResult = new Vector(rnd1.NextDouble(), rnd1.NextDouble());
            vectorResult.Normalize();
            xValues[0] = vectorResult.X;
            xValues[1] = vectorResult.Y;

            tValues[0] = Math.Atan2(xValues[1],xValues[0]);


            /*
            xValues[0] = rnd1.NextDouble();
            tValues[0] = Math.Cos(xValues[0]);
            */

            double[] yValues = ComputeOutputs(xValues); 
            UpdateWeights(tValues, learnRate, momentum);
            prevLoss = loss;
            loss = tValues[0] - yValues[0];
            //if (epoch % 100 == 0) 
            {
                /*
                Console.Write("epoch = " + epoch.ToString().PadLeft(5) + " current outputs = ");
                BackPropProgram.ShowVector(yValues, numOutput, 4, true); 
                */

                Console.Write("\n epoch = " + epoch.ToString().PadLeft(5) +"  x:"+xValues[0].ToString("F3")+" predicted:"+yValues[0].ToString("F3")+"  out:"+outputs[0].ToString("F3")+"   truth:"+tValues[0].ToString("F3")+"    loss = "+loss.ToString("F3"));

            }
            ++epoch;

            if(Double.IsNaN(yValues[0]))
                break;

            if (Math.Abs(loss) < 0.0001 &&  Math.Abs(prevLoss) < 0.0001)
            {
                break;
            }
                    
        } // Find loop.
                    
    }


    private double DReLU(double x)
    {
        //return Math.Max(0,1);// x < 0 ? 0 : x;
        return x < 0 ? 0 : 1;
    }
                    
    private void UpdateWeights(double[] tValues, double learnRate, double momentum) 
    {
        // Assumes that SetWeights and ComputeOutputs have been called.
        if (tValues.Length != numOutput)
            throw new Exception("target values not same Length as output in UpdateWeights");

        // 1. Compute output gradients. Assumes softmax.
        for (int i = 0; i < oGrads.Length; ++i) {
            //double derivative = (1 - outputs[i]) * outputs[i]; // Derivative of softmax is y(1- y).
            double derivative = DReLU(outputs[i]); // Derivative of relu 
            //double derivative = DLogistic(outputs[i]); // Derivative of relu 
            oGrads[i] = derivative * (tValues[i] - outputs[i]); // oGrad = (1 - O)(O) * (T-O)
        }

        // 2. Compute hidden gradients. Assumes tanh!
        for (int i = 0; i < h2Grads.Length; ++i)
        {
            //double derivative = (1 - hOutputs[i]) * (1 + hOutputs[i]); // f' of tanh is (1-y)(1+y).
            double derivative = DReLU(h2Outputs[i]); // Derivative of relu 
            //double derivative = DLogistic(h2Outputs[i]); // Derivative of relu 
            double sum = 0.0;
            for (int j = 0; j < numOutput; ++j) // Each hidden delta is the sum of numOutputterms.
                sum += oGrads[j] * h2oWeights[i][j]; // Each downstream gradient * outgoing weight.
            h2Grads[i] = derivative * sum; // hGrad = (1-O)(1+O) * Sum(oGrads*oWts)
        }


        // 2.1 Compute hidden gradients. Assumes tanh!
        for (int i = 0; i < hGrads.Length; ++i)
        {
            //double derivative = (1 - hOutputs[i]) * (1 + hOutputs[i]); // f' of tanh is (1-y)(1+y).
            double derivative = DReLU(hOutputs[i]); // Derivative of relu 
            //double derivative = DLogistic(hOutputs[i]); // Derivative of relu 
            double sum = 0.0;
            for (int j = 0; j < numHidden2; ++j) // Each hidden delta is the sum of numOutputterms.
                sum += h2Grads[j] * hoWeights[i][j]; // Each downstream gradient * outgoing weight.
            hGrads[i] = derivative * sum; // hGrad = (1-O)(1+O) * Sum(oGrads*oWts)
        }

        // 3. Update input to hidden weights.
        for (int i = 0; i < ihWeights.Length; ++i)
        {
            for (int j = 0; j < ihWeights[i].Length; ++j)
            {
                double delta = learnRate * hGrads[j] * inputs[i]; 
                ihWeights[i][j] += delta; // Update.
                ihWeights[i][j] += momentum * ihPrevWeightsDelta[i][j]; // Add momentum factor.
                ihPrevWeightsDelta[i][j] = delta; // Save the delta for next time.
            }
        }

        // 4. Update hidden biases.
        for (int i = 0; i < hBiases.Length; ++i)
        {
            double delta = learnRate * hGrads[i] * 1.0; // The 1.0 is a dummy value; it could be left out.
            hBiases[i] += delta;
            hBiases[i] += momentum * hPrevBiasesDelta[i];
            hPrevBiasesDelta[i] = delta; // Save delta.
        }




        // 3.1 Update hidden to hidden2 weights.
        for (int i = 0; i < hoWeights.Length; ++i)
        { 
            for (int j = 0; j < hoWeights[i].Length; ++j) 
            {
                double delta = learnRate * h2Grads[j] * hOutputs[i]; 
                hoWeights[i][j] += delta; // Update.
                hoWeights[i][j] += momentum * hoPrevWeightsDelta[i][j]; // Add momentum factor.
                hoPrevWeightsDelta[i][j] = delta; // Save the delta for next time.
            }
        }

        // 4.1 Update hidden2 biases.
        for (int i = 0; i < h2Biases.Length; ++i)
        {
            double delta = learnRate * h2Grads[i] * 1.0; // The 1.0 is a dummy value; it could be left out.
            h2Biases[i] += delta;
            h2Biases[i] += momentum * h2PrevBiasesDelta[i];
            h2PrevBiasesDelta[i] = delta; // Save delta.
        }





        // 5. Update hidden2 to output weights.
        for (int i = 0; i < h2oWeights.Length; ++i)
        { 
            for (int j = 0; j < h2oWeights[i].Length; ++j) 
            {
                double delta = learnRate * oGrads[j] * h2Outputs[i]; 
                h2oWeights[i][j] += delta;
                h2oWeights[i][j] += momentum * h2oPrevWeightsDelta[i][j];
                h2oPrevWeightsDelta[i][j] = delta; // Save delta.
            }
        }



        /*
        // 4.2 Update hidden biases.
        for (int i = 0; i < hBiases.Length; ++i)
        {
            double delta = learnRate * hGrads[i] * 1.0; // The 1.0 is a dummy value; it could be left out.
            hBiases[i] += delta;
            hBiases[i] += momentum * hPrevBiasesDelta[i];
            hPrevBiasesDelta[i] = delta; // Save delta.
        }

        // 5.2 Update hidden to output weights.
        for (int i = 0; i < hoWeights.Length; ++i)
        { for (int j = 0; j < hoWeights[i].Length; ++j) 
            {
                double delta = learnRate * oGrads[j] * hOutputs[i]; 
                hoWeights[i][j] += delta;
                hoWeights[i][j] += momentum * hoPrevWeightsDelta[i][j];
                hoPrevWeightsDelta[i][j] = delta; // Save delta.
            }
        }
        */


        // 6. Update output biases.
        for (int i = 0; i < oBiases.Length; ++i)
        {
            double delta = learnRate * oGrads[i] * 1.0;
            oBiases[i] += delta;
            oBiases[i] += momentum * oPrevBiasesDelta[i];
            oPrevBiasesDelta[i] = delta; // Save delta.
        }
    } // UpdateWeights
} // NN class
