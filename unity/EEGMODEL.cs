using UnityEngine;
using Unity.InferenceEngine;

public class EEGMODEL : MonoBehaviour
{
    private Model runtimeModel;
    private Worker worker;

    void Start()
    {
        runtimeModel = ModelLoader.Load(Application.streamingAssetsPath + "/eegnet_embedded.sentis");
        worker = new Worker(runtimeModel, BackendType.CPU);
        Debug.Log("EEG Model loaded successfully.");
    }

    public int Classify(float[] eegData)
    {
        using Tensor<float> inputTensor = new Tensor<float>(new TensorShape(1, 1, 64, 129), eegData);

        worker.Schedule(inputTensor);

        using Tensor<float> output = (worker.PeekOutput(runtimeModel.outputs[0].name) as Tensor<float>).ReadbackAndClone();

        int predictedClass = 0;
        float maxVal = float.MinValue;
        for (int i = 0; i < 3; i++)
        {
            if (output[i] > maxVal)
            {
                maxVal = output[i];
                predictedClass = i;
            }
        }
        return predictedClass;
    }

    void OnDestroy()
    {
        worker?.Dispose();
    }
}