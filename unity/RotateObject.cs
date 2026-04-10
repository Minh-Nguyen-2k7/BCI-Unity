using UnityEngine;

public class RotateObject : MonoBehaviour
{
    public EEGStreamer streamer;
    public float rotationSpeed = 90f; // degrees per second

    void Update()
    {
        if (streamer == null) return;

        switch (streamer.latestPrediction)
        {
            case 1: // Left
                transform.Rotate(Vector3.up, -rotationSpeed * Time.deltaTime);
                break;
            case 2: // Right
                transform.Rotate(Vector3.up, rotationSpeed * Time.deltaTime);
                break;
            // case 0 (Rest): do nothing
        }
    }
}
