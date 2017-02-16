//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTKLibraryCSEvalExamples.cs -- Examples for using CNTK Library C# Eval API.
//

using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using CNTK;

namespace CNTKLibraryCSEvalExamples
{
    public class CNTKLibraryManagedExamples
    {
        // 
        // The example shows
        // - how to load model.
        // - how to prepare input data for a single sample.
        // - how to prepare input and output data map.
        // - how to evaluate a model.
        // - how to retrieve evaluation result and retrieve output data in dense format.
        //
        public static void EvaluationSingleImage(DeviceDescriptor device)
        {
            var inputDataMap = new Dictionary<Variable, Value>();

            // Load the model.
            // The model resnet20.dnn is trained by <CNTK>/Examples/Image/Classification/ResNet/Python/Models/TrainResNet_CIFAR10.py
            // Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
            string modelFilePath = "resnet20.dnn";
            ThrowIfFileNotExist(modelFilePath, string.Format("Error: The model '{0}' does not exist. Please follow instructions in README.md in <CNTK>/Examples/Image/Classification/ResNet to create the model.", modelFilePath));
            Function modelFunc = Function.LoadModel(modelFilePath, device);

            // Get output variable based on name
            // Variable outputVar = modelFunc.Outputs.Where(variable => string.Equals(variable.Name, outputName)).Single();
            Variable outputVar = modelFunc.Output;

            // Get input variable. The model has only one single input.
            // The same way described above for output variable can be used here to get input variable by name.
            Variable inputVar = modelFunc.Arguments.Single();
            var outputDataMap = new Dictionary<Variable, Value>();
            Value inputVal, outputVal;
            List<List<float>> outputBuffer;

            // Get shape data for the input variable
            NDShape inputShape = inputVar.Shape;
            uint imageWidth = inputShape[0];
            uint imageHeight = inputShape[1];
            uint imageChannels = inputShape[2];
            uint imageSize = inputShape.TotalSize;

            Console.WriteLine("Evaluate single image");

            // Image preprocessing to match input requirements of the model.
            Bitmap bmp = new Bitmap(Bitmap.FromFile("00000.png"));
            var resized = bmp.Resize((int)imageWidth, (int)imageHeight, true);
            List<float> resizedCHW = resized.ParallelExtractCHW();

            // Create input data map
            inputVal = Value.CreateBatch(inputVar.Shape, resizedCHW, device);
            inputDataMap.Add(inputVar, inputVal);

            // Create ouput data map. Using null as Value to indicate using system allocated memory.
            // Alternatively, create a Value object and add it to the data map.
            outputDataMap.Add(outputVar, null);

            // Start evaluation on the device
            modelFunc.Evaluate(inputDataMap, outputDataMap, device);

            // Get evaluate result as dense output
            outputBuffer = new List<List<float>>();
            outputVal = outputDataMap[outputVar];
            outputVal.CopyVariableValueTo(outputVar, outputBuffer);

            PrintOutput(outputVar.Shape.TotalSize, outputBuffer);
        }

        //
        // The example shows
        // - how to load model.
        // - how to prepare input data for a batch of samples.
        // - how to prepare input and output data map.
        // - how to evaluate a model.
        // - how to retrieve evaluation result and retrieve output data in dense format.
        //
        public static void EvaluationBatchOfImages(DeviceDescriptor device)
        {
            var inputDataMap = new Dictionary<Variable, Value>();

            // Load the model.
            // The model resnet20.dnn is trained by <CNTK>/Examples/Image/Classification/ResNet/Python/Models/TrainResNet_CIFAR10.py
            // Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
            string modelFilePath = "resnet20.dnn";
            ThrowIfFileNotExist(modelFilePath, string.Format("Error: The model '{0}' does not exist. Please follow instructions in README.md in <CNTK>/Examples/Image/Classification/ResNet to create the model.", modelFilePath));
            Function modelFunc = Function.LoadModel(modelFilePath, device);

            // Get output variable based on name
            Variable outputVar = modelFunc.Output;

            // Get input variable. The model has only one single input.
            // The same way described above for output variable can be used here to get input variable by name.
            Variable inputVar = modelFunc.Arguments.Single();
            var outputDataMap = new Dictionary<Variable, Value>();
            Value inputVal, outputVal;
            List<List<float>> outputBuffer;

            // Get shape data for the input variable
            NDShape inputShape = inputVar.Shape;
            uint imageWidth = inputShape[0];
            uint imageHeight = inputShape[1];
            uint imageChannels = inputShape[2];
            uint imageSize = inputShape.TotalSize;

            Console.WriteLine("\nEvaluate batch of images");

            Bitmap bmp, resized;
            List<float> resizedCHW;

            var fileList = new List<string>() { "00000.png", "00001.png", "00002.png" };
            var seqData = new List<float>();
            for (int sampleIndex = 0; sampleIndex < fileList.Count; sampleIndex++)
            {
                bmp = new Bitmap(Bitmap.FromFile(fileList[sampleIndex]));
                resized = bmp.Resize((int)imageWidth, (int)imageHeight, true);
                resizedCHW = resized.ParallelExtractCHW();
                // Aadd this sample to the data buffer.
                seqData.AddRange(resizedCHW);
            }

            // Create Value for the batch data.
            inputVal = Value.CreateBatch(inputVar.Shape, seqData, device);

            // Create input data map.
            inputDataMap.Add(inputVar, inputVal);

            // Create ouput data map. Using null as Value to indicate using system allocated memory.
            // Alternatively, create a Value object and add it to the data map.
            outputDataMap.Add(outputVar, null);

            // Evaluate the model against the batch input
            modelFunc.Evaluate(inputDataMap, outputDataMap, device);

            // Retrieve the evaluation result.
            outputBuffer = new List<List<float>>();
            outputVal = outputDataMap[outputVar];
            outputVal.CopyVariableValueTo(outputVar, outputBuffer);

            // Output result
            PrintOutput(outputVar.Shape.TotalSize, outputBuffer);
        }

        //
        // The example shows
        // - how to load model.
        // - how to prepare input data as sequence.
        // - how to retrieve input and output variables by name.
        // - how to prepare input and output data map.
        // - how to evaluate a model.
        // - how to retrieve evaluation result and retrieve output data in the one-hot vector format.
        //
        public static void EvaluationSingleSequenceUsingOneHot(DeviceDescriptor device)
        {
            // The model atis.dnn is trained by <CNTK>/Examples/LanguageUnderstanding/ATIS/Python/LanguageUnderstanding.py
            // Please see README.md in <CNTK>/Examples/LanguageUnderstanding/ATIS about how to train the model.
            string modelFilePath = "atis.dnn";
            ThrowIfFileNotExist(modelFilePath, string.Format("Error: The model '{0}' does not exist. Please follow instructions in README.md in <CNTK>/Examples/LanguageUnderstanding/ATIS to create the model.", modelFilePath));
            Function myFunc = Function.LoadModel(modelFilePath, device);

            string vocabFile = "ATIS.vocab";
            string labelFile = "ATIS.label";
            ThrowIfFileNotExist(vocabFile, string.Format("Error: The file '{0}' does not exist. Please copy it from <CNTK>/Examples/LanguageUnderstanding/ATIS/Data to the output directory.", vocabFile));
            ThrowIfFileNotExist(labelFile, string.Format("Error: The file '{0}' does not exist. Please copy it from <CNTK>/Examples/LanguageUnderstanding/ATIS/Data to the output directory.", labelFile));
            var vocabToIndex = buildVocabIndex(vocabFile);
            var indexToVocab = buildInvVocabIndex(labelFile);

            Console.WriteLine("Evaluate single sequence using one-hot vector");

            // Get input variable
            var inputVar = myFunc.Arguments.Single();

            uint vocabSize = inputVar.Shape.TotalSize;

            // Use case 1: Evalaute a single sequence using OneHot vector as input.
            var inputSentence = "BOS i would like to find a flight from charlotte to las vegas that makes a stop in st. louis EOS";
            // Build input data for one sequence 
            var seqData = new List<uint>();
            // SeqStartFlagBatch is used to indicate whether this sequence is a new sequence (true) or concatenating the previous sequence (false).
            var seqStartFlag = true;
            // Get the index of each word in the sentence.
            string[] substring = inputSentence.Split(' ');
            foreach (var str in substring)
            {
                // Get the index of the word
                var index = vocabToIndex[str];
                // Add the sample to the sequence
                seqData.Add(index);
            }

            // Create input value using OneHot vector data.
            var inputValue = Value.CreateSequence<float>(vocabSize, seqData, seqStartFlag, device);

            // Build input data map.
            var inputDataMap = new Dictionary<Variable, Value>();
            inputDataMap.Add(inputVar, inputValue);

            // Prepare output
            Variable outputVar = myFunc.Output;

            // Create ouput data map. Using null as Value to indicate using system allocated memory.
            var outputDataMap = new Dictionary<Variable, Value>();
            outputDataMap.Add(outputVar, null);

            // Evalaute the model.
            myFunc.Evaluate(inputDataMap, outputDataMap, device);

            // Get output result
            var outputData = new List<List<uint>>();
            Value outputVal = outputDataMap[outputVar];
            outputVal.CopyVariableValueTo(outputVar, outputData);

            // output the result
            var numOfElementsInSample = vocabSize;
            uint seqNo = 0;
            foreach (var seq in outputData)
            {
                Console.Write("Seq=" + seqNo + ":");
                foreach (var index in seq)
                {
                    // get the word based on index
                    Console.Write(indexToVocab[index]);
                }
                Console.WriteLine();
                // next sequence.
                seqNo++;
            }
        }

        //
        // The example shows
        // - how to load model.
        // - how to prepare input data as batch of sequences with variable length.
        // - how to retrieve input and output variables by name.
        // - how to prepare input and output data map.
        // - how to evaluate a model.
        // - how to retrieve evaluation result and retrieve output data in the one-hot vector format.
        //
        public static void EvaluationBatchOfSequencesUsingOneHot(DeviceDescriptor device)
        {
            // The model atis.dnn is trained by <CNTK>/Examples/LanguageUnderstanding/ATIS/Python/LanguageUnderstanding.py
            // Please see README.md in <CNTK>/Examples/LanguageUnderstanding/ATIS about how to train the model.
            string modelFilePath = "atis.dnn";
            ThrowIfFileNotExist(modelFilePath, string.Format("Error: The model '{0}' does not exist. Please follow instructions in README.md in <CNTK>/Examples/LanguageUnderstanding/ATIS to create the model.", modelFilePath));
            Function myFunc = Function.LoadModel(modelFilePath, device);

            string vocabFile = "ATIS.vocab";
            string labelFile = "ATIS.label";
            ThrowIfFileNotExist(vocabFile, string.Format("Error: The file '{0}' does not exist. Please copy it from <CNTK>/Examples/LanguageUnderstanding/ATIS/Data to the output directory.", vocabFile));
            ThrowIfFileNotExist(labelFile, string.Format("Error: The file '{0}' does not exist. Please copy it from <CNTK>/Examples/LanguageUnderstanding/ATIS/Data to the output directory.", labelFile));
            var vocabToIndex = buildVocabIndex(vocabFile);
            var indexToVocab = buildInvVocabIndex(labelFile);

            Console.WriteLine("Evaluate batch of sequences with variable length using one-hot vector");

            // Get input variable 
            const string inputName = "features";
            var inputVar = myFunc.Arguments.Where(variable => string.Equals(variable.Name, inputName)).Single();

            uint vocabSize = inputVar.Shape.TotalSize;

            // Prepare the input data. 
            // Each sample is represented by an index to the onehot vector, so the index of the non-zero value of each sample is saved in the inner list.
            // The outer list represents sequences contained in the batch.
            var inputBatch = new List<List<uint>>();
            // SeqStartFlagBatch is used to indicate whether this sequence is a new sequence (true) or concatenating the previous sequence (false).
            var seqStartFlagBatch = new List<bool>();

            var inputSentences = new List<string>() { 
                "BOS i would like to find a flight from charlotte to las vegas that makes a stop in st. louis EOS",
                "BOS I want to book a flight from New York to Seattle EOS"
            };

            List<uint> seqData;
            string[] substring;
            int numOfSequences = inputSentences.Count;
            for (int seqIndex = 0; seqIndex < numOfSequences; seqIndex++)
            {
                // The input for one sequence 
                seqData = new List<uint>();
                // Get the index of each word in the sentence.
                substring = inputSentences[seqIndex].Split(' ');
                foreach (var str in substring)
                {
                    var index = vocabToIndex[str];
                    seqData.Add(index);
                }
                inputBatch.Add(seqData);
                seqStartFlagBatch.Add(true);
            }

            // Create the Value representing the batch data.
            var inputValue = Value.CreateBatchOfSequences<float>(vocabSize, inputBatch, seqStartFlagBatch, DeviceDescriptor.CPUDevice);

            // Build input data map.
            var inputDataMap = new Dictionary<Variable, Value>();
            inputDataMap.Add(inputVar, inputValue);

            // Prepare output
            const string outputName = "out.z";
            Variable outputVar = myFunc.Outputs.Where(variable => string.Equals(variable.Name, outputName)).Single();

            // Create ouput data map. Using null as Value to indicate using system allocated memory.
            var outputDataMap = new Dictionary<Variable, Value>();
            outputDataMap.Add(outputVar, null);

            // Evalaute the model
            myFunc.Evaluate(inputDataMap, outputDataMap, device);

            // Get evaluation result.
            var outputData = new List<List<uint>>();
            var outputVal = outputDataMap[outputVar];
            outputVal.CopyVariableValueTo(outputVar, outputData);

            // output the result
            var numOfElementsInSample = vocabSize;
            uint seqNo = 0;
            foreach (var seq in outputData)
            {
                Console.Write("Seq=" + seqNo + ":");
                foreach (var index in seq)
                {
                    // get the word based on index
                    Console.Write(indexToVocab[index]);
                }
                Console.WriteLine();
                // next sequence.
                seqNo++;
            }
        }

        /// <summary>
        /// Checks whether the file exists. If not, write the error message on the console and throw FileNotFoundException.
        /// </summary>
        /// <param name="filePath">The file to check.</param>
        /// <param name="errorMsg">The message to write on console if the file does not exist.</param>
        private static void ThrowIfFileNotExist(string filePath, string errorMsg)
        {
            if (!File.Exists(filePath))
            {
                if (!string.IsNullOrEmpty(errorMsg))
                {
                    Console.WriteLine(errorMsg);
                }
                throw new FileNotFoundException(string.Format("File '{0}' not found.", filePath));
            }
        }
        private static void PrintOutput<T>(uint sampleSize, List<List<T>> outputBuffer)
        {
            Console.WriteLine("The number of sequences in the batch: " + outputBuffer.Count);
            int seqNo = 0;
            uint outputSampleSize = sampleSize;
            foreach (var seq in outputBuffer)
            {
                Console.WriteLine(String.Format("Sequence {0} contains {1} samples.", seqNo++, seq.Count / outputSampleSize));
                uint i = 0;
                uint sampleNo = 0;
                foreach (var element in seq)
                {
                    if (i++ % outputSampleSize == 0)
                    {
                        Console.Write(String.Format("    sample {0}: ", sampleNo));
                    }
                    Console.Write(element);
                    if (i % outputSampleSize == 0)
                    {
                        Console.WriteLine(".");
                        sampleNo++;
                    }
                    else
                    {
                        Console.Write(",");
                    }
                }
            }
        }

        private static Dictionary<string, uint> buildVocabIndex(string filePath)
        {
            var vocab = new Dictionary<string,uint>();

            string[] lines = File.ReadAllLines(filePath);
            for (uint idx = 0; idx < (uint)lines.Count(); idx++)
                vocab.Add(lines[idx], idx);

            return vocab;
        }

        private static string[] buildInvVocabIndex(string filePath)
        {
            return File.ReadAllLines(filePath);
        }
    }
}
