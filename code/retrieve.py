import influxdb_client
import numpy as np # Import numpy for the final conversion

# --- Configuration ---
INFLUXDB_URL = "https://us-east-1-1.aws.cloud2.influxdata.com"
INFLUXDB_TOKEN = "xyegPQg3ldYHuoN2brTlLOwLEz6671vRfciLo6IlHZSMzP3Yn7CaDbI6ghw6wgdLmcgQuXWDAH5E203ypw_sRg=="
INFLUXDB_ORG = "project_1"
INFLUXDB_BUCKET = "test_1"
MEASUREMENT_NAME = "ecg"
FIELD_NAME = "value"

def retrieve_values_as_array(time_range_days=30):
    """
    Connects to InfluxDB and retrieves all non-null values from a single field,
    returning them as a Python list.
    """
    print("Connecting to InfluxDB to retrieve all values...")
    try:
        client = influxdb_client.InfluxDBClient(
            url=INFLUXDB_URL,
            token=INFLUXDB_TOKEN,
            org=INFLUXDB_ORG
        )
        query_api = client.query_api()
    except Exception as e:
        print(f"Error connecting to InfluxDB: {e}")
        return None

    # --- Modified Flux Query ---
    # 1. Added 'exists r._value' to ensure we only get rows where the value is not null.
    flux_query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: -{time_range_days}d)
      |> filter(fn: (r) => r._measurement == "{MEASUREMENT_NAME}")
      |> filter(fn: (r) => r._field == "{FIELD_NAME}")
      |> filter(fn: (r) => exists r._value) // <-- IMPORTANT: This filters out any null values
      |> sort(columns: ["_time"])
    '''

    print(f"Executing query to retrieve all non-null values from the last {time_range_days} days...")
    ecg_values = []
    try:
        tables = query_api.query(flux_query, org=INFLUXDB_ORG)
        
        # Loop through the results and append only the value to our list
        for table in tables:
            for record in table.records:
                ecg_values.append(record.get_value())
        print(ecg_values)

    except Exception as e:
        print(f"Error querying InfluxDB: {e}")
        client.close()
        return None

    client.close()
    return ecg_values


# --- Main execution block ---
if __name__ == "__main__":
    # 1. Fetch the data as a simple Python list
    all_values = retrieve_values_as_array(time_range_days=30)

    if all_values is not None:
        print("\n-----------------------------------------")
        print(f"Successfully retrieved a total of {len(all_values)} values.")
        print("-----------------------------------------")

        # 2. Print the entire array of values
        print("\nComplete array of values:")
        print(all_values)

        # 3. (Optional but recommended for AI) Convert the list to a NumPy array
        values_np = np.array(all_values)
        print("\nValues as a NumPy array:")
        print(values_np)
        print(f"Shape of NumPy array: {values_np.shape}")
    else:
        print("Failed to retrieve data.")