import influxdb_client
from influxdb_client import Point
from influxdb_client.client.write_api import SYNCHRONOUS

# --- Configuration (Copied from your C++ code) ---
INFLUXDB_URL = "https://us-east-1-1.aws.cloud2.influxdata.com"
INFLUXDB_TOKEN = "149JOK3dJC7_ehnHD63jNdmONRENjZ9VkZJ-QM5cK9_Ec5TUfTFzPtYLlfkwKR83K4wCUZu-g1r3XG4uebsfug=="
INFLUXDB_ORG = "project_1"
INFLUXDB_BUCKET = "prediction" # We write the prediction to the same bucket

def write_prediction(prediction_value, session_id=None):
    """
    Connects to InfluxDB and writes a prediction value to a new measurement.

    Args:
        prediction_value (str, float, int): The value your AI model predicted.
                                            (e.g., "Normal", "MI", or a probability score).
        session_id (str, optional): The unique ID of the ECG recording this
                                    prediction is for. Highly recommended for linking data.

    Returns:
        bool: True if the write was successful, False otherwise.
    """
    print("Connecting to InfluxDB to write prediction...")
    try:
        client = influxdb_client.InfluxDBClient(
            url=INFLUXDB_URL,
            token=INFLUXDB_TOKEN,
            org=INFLUXDB_ORG
        )
        # Use SYNCHRONOUS mode for simple, direct writes.
        write_api = client.write_api(write_options=SYNCHRONOUS)
    except Exception as e:
        print(f"Error connecting to InfluxDB: {e}")
        return False

    # --- Create the Data Point ---
    # A 'Point' is the object used to structure a single row of data for InfluxDB.
    
    # 1. Set the measurement name to "prediction"
    point = Point("prediction") 
    
    # 2. (Recommended) Add a tag to link this prediction to the original data
    if session_id:
        point.tag("session_id", session_id)
        
    # 3. Add the field "predicted" with the value from your model
    point.field("predicted", prediction_value)

    try:
        print(f"Writing point to InfluxDB: Measurement='prediction', Field='predicted', Value='{prediction_value}'")
        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
        print("Write successful.")
        
    except Exception as e:
        print(f"Error writing to InfluxDB: {e}")
        client.close()
        return False

    client.close()
    return True

# --- Main execution block ---
if __name__ == "__main__":
    # --- EXAMPLE USAGE ---
    
    # 1. This is the output from your AI model.
    #    It can be a string (like a diagnosis) or a number (like a probability).
    model_output = "Myocardial Infarction" 
    
    # 2. This should be the unique ID from the ECG data you just analyzed.
    #    This allows you to find exactly which ECG recording this prediction belongs to.
    original_data_session_id = "1678886400" # Example session ID

    # 3. Call the function to push the prediction to InfluxDB.
    success = write_prediction(model_output, session_id=original_data_session_id)

    if success:
        print("\nPrediction was successfully stored in InfluxDB.")
    else:
        print("\nFailed to store prediction.")