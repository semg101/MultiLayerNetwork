using System;

namespace MultiLayerNetwork
{

    public class MultiLayerNetwork
    {
        public List<Layer> Layers { get; set; }
        public List<TrainingSample> TrainingSamples { get; set; }
        public int HiddenUnits{ get; set; }
        public int OutputUnits{ get; set; }
        public double LearningRate { get; set; }
        private double _maxError;

        public MultiLayerNetwork(IEnumerable<TrainingSample> trainingSamples, int inputs, inthiddenUnits, int outputs, double learningRate)
        {
            Layers = new List<Layer>(); 
            TrainingSamples = new List<TrainingSample>(trainingSamples); 
            LearningRate = learningRate; 
            HiddenUnits = hiddenUnits; 
            OutputUnits = outputs; 
            CreateLayers(inputs);
        }

        private void CreateLayers(int inputs)
        {
            Layers.Add(new Layer(HiddenUnits, TrainingSamples, LearningRate, inputs, TypeofLayer.Hidden)); 
            Layers.Add(new Layer(OutputUnits, TrainingSamples, LearningRate, HiddenUnits, TypeofLayer.OutPut));
        }

        public List<double> PredictSet(IEnumerable<double[]> objects) 
        { 
            var result = new List<double>(); 
            foreach (varobj in objects) 
                result.Add(Predict(obj)); 
            return result; 
        }

        public Layer OutPutLayer 
        { 
            get { returnLayers.Last(); } 
        }

        public Layer HiddenLayer 
        { 
            get { returnLayers.First(); } 
        }
    }

    public class Layer
    {
        
        public List<SigmoidUnit> Units { get; set; }
        public TypeofLayer Type { get; set; }

        public Layer(int number, List<TrainingSample> trainingSamples, double learningRate, int inputs, TypeofLayer typeofLayer)
        {
            Units = new List<SigmoidUnit>(); 
            Type = typeofLayer; 
            for (var i = 0; i < number; i++) 
                Units.Add(new SigmoidUnit(trainingSamples, inputs, learningRate));
        }
    }

    public enum TypeofLayer { Hidden, OutPut }

    public class SigmoidUnit : SingleNeuralNetwork
    {
        public double ActivationValue { get; set; }
        public double ErrorTerm { get; set; }
        public SigmoidUnit(IEnumerable<TrainingSample> trainingSamples, int inputs, double learningRate) : base(trainingSamples, inputs, learningRate) { }

        public override double Predict(double[] features)
        {
            var result = 0.0;
            for (var i = 0; i < features.Length; i++)
                result += features[i] * Weights[i];
            return ActivationValue = 1 / (1 + Math.Pow(Math.E, -result));
        }

        public void Training()
        {
            _maxError = double.MaxValue;
            while (Math.Abs(_maxError) > .001)
            {
                foreach (vartrainingSample in TrainingSamples)
                {
                    Predict(trainingSample.Features);

                    // Error term for output layer ...
                    for (var i = 0; i < OutPutLayer.Units.Count; i++)
                    {
                        OutPutLayer.Units[i].ErrorTerm = FunctionDerivative(OutPutLayer.Units[i].ActivationValue, TypeFunction.Sigmoid) * (trainingSample.Classifications[i] - OutPutLayer.Units[i].ActivationValue);

                        // Error term for hidden layer ...
                        for (var i = 0; i < HiddenLayer.Units.Count; i++)
                        {
                            var outputUnitWeights = OutPutLayer.Units.Select(u => u.Weights[i]).ToList();
                            var product = (from j in Enumerable.Range(0, outputUnitWeights.Count) select outputUnitWeights[j] * OutPutLayer.Units[j].ErrorTerm).Sum();
                            HiddenLayer.Units[i].ErrorTerm = FunctionDerivative(HiddenLayer.Units[i].ActivationValue, TypeFunction.Sigmoid) * product;
                        }

                        UpdateWeight(trainingSample.Features, OutPutLayer);
                        UpdateWeight(trainingSample.Features, HiddenLayer);
                        _maxError = OutPutLayer.Units.Max(u => Math.Abs(u.ErrorTerm));
                    }
                }
            }
        }

        private double FunctionDerivative(double v, TypeFunction function)
        {
            switch (function)
            {
                case TypeFunction.Sigmoid:
                    return v * (1 - v);
                case TypeFunction.Tanh:
                    return 1 - Math.Pow(v, 2);
                case TypeFunction.ReLu:
                    return Math.Max(0, v);
                default:
                    return 0;
            }
        }

        public enum TypeFunction{Sigmoid, Tanh, ReLu }

        private void UpdateWeight(double[] features, Layer layer)
        {
            var activationValues = layer.Type == TypeofLayer.Hidden ? features : HiddenLayer.Units.Select(u => u.ActivationValue).ToArray();

            foreach (var unit in layer.Units)
            {
                for (var i = 0; i < unit.Weights.Count; i++)
                    unit.Weights[i] += LearningRate * unit.ErrorTerm * activationValues[i];
            }
        }
    }

    public class TanhUnit : SingleNeuralNetwork
    {
        public double ActivationValue { get; set; }
        public double ErrorTerm { get; set; }
        public TanhUnit(IEnumerable<TrainingSample> training Samples, int inputs, double learningRate) : base(trainingSamples, inputs, learningRate) { }

        public override double Predict(double[] features)
        {
            var result = 0.0;
            for (var i = 0; i < features.Length; i++)
                result += features[i] * Weights[i];
            ActivationValue = Math.Tanh(result);
            return ActivationValue;
        }

        public double Predict(double[] features)
        {

        }
    }

    public class ReLu : SingleNeuralNetwork
    {
        public double ActivationValue { get; set; }
        public double ErrorTerm { get; set; }
        public ReLu(IEnumerable<TrainingSample> trainingSamples, int inputs, double learningRate) : base(trainingSamples, inputs, learningRate) { }

        public override double Predict(double[] features) 
        { 
            var result = 0.0; 
            for (var i = 0; i < features.Length; i++) 
                result += features[i] * Weights[i]; 
            return Math.Max(0, result); 
        }
    }





    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
        }
    }
}
