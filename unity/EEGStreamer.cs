using UnityEngine;
using System.IO;
using System.Collections;

public class EEGStreamer : MonoBehaviour
{
    private string csvPath;
    private StreamReader reader;
    private int totalEpochs = 0;
    private int correct = 0;

    // 0 = Rest, 1 = Left, 2 = Right
    public int latestPrediction = 0;
    
    // Reference to your classifier
    public EEGMODEL classifier;
    
    // How fast to stream (seconds between each epoch)
    public float streamInterval = 0.8f;
    
    string[] classNames = { "Rest", "Left", "Right" };

    void Start()
    {
        csvPath = Path.Combine(Application.streamingAssetsPath, "eeg_test_stream.csv");
        
        if (!File.Exists(csvPath))
        {
            Debug.LogError("CSV file not found at: " + csvPath);
            return;
        }
        
        reader = new StreamReader(csvPath);
        reader.ReadLine();
        
        Debug.Log("✅ CSV loaded. Starting stream...");
        StartCoroutine(StreamEpochs());
    }

    IEnumerator StreamEpochs()
    {
        yield return new WaitForSeconds(1.0f);

        while (!reader.EndOfStream)
        {
            string line = reader.ReadLine();
            if (string.IsNullOrEmpty(line)) continue;
            
            string[] values = line.Split(',');
            
            float[] eegData = new float[8256];
            for (int i = 0; i < 8256; i++)
            {
                eegData[i] = float.Parse(values[i]);
            }
            
            int trueLabel = Mathf.RoundToInt(float.Parse(values[8256]));
            
            int predicted = classifier.Classify(eegData);
            latestPrediction = predicted;
            
            totalEpochs++;
            if (predicted == trueLabel) correct++;
            
            float accuracy = (float)correct / totalEpochs * 100f;
            
            Debug.Log($"Epoch {totalEpochs} | " +
                      $"Predicted: {classNames[predicted]} | " +
                      $"Actual: {classNames[trueLabel]} | " +
                      $"Accuracy: {accuracy:F2}%");
            
            yield return new WaitForSeconds(streamInterval);
        }
        
        reader.Close();
        Debug.Log($"✅ Stream complete! Final Accuracy: {(float)correct/totalEpochs*100f:F2}%");
    }

    void OnDestroy()
    {
        reader?.Close();
    }
}