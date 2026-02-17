from src.database.influx import DatabaseHandler
import os

def main():
    try:
        db = DatabaseHandler()
        df = db.fetch_data(measurement="EM-300", range_start="0")
        
        if not df.empty:
            output_dir = "data/raw"
            os.makedirs(output_dir, exist_ok=True)
            
            file_path = os.path.join(output_dir, "sensor_data_raw.csv")
            df.to_csv(file_path, index=False)
            
            print(f"Data fetched and saved to {file_path}")
            # print(df.head())
        else:
            print("No data found.")
            
    except Exception as e:
        print(f"An error occurred in main: {e}")
    finally:
        if 'db' in locals():
            db.close()

if __name__ == "__main__":
    main()