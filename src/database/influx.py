import os
import pandas as pd
import influxdb_client
from dotenv import load_dotenv

load_dotenv()

class DatabaseHandler:
    def __init__(self):
        self.url = os.getenv("INFLUXDB_URL")
        self.token = os.getenv("INFLUXDB_TOKEN")
        self.org = os.getenv("INFLUXDB_ORG")
        self.bucket = os.getenv("INFLUXDB_BUCKET")

        if not all([self.url, self.token, self.org, self.bucket]):
            raise ValueError("Missing InfluxDB configuration in .env file")

        self.client = influxdb_client.InfluxDBClient(
            url=self.url, 
            token=self.token, 
            org=self.org
        )
        self.query_api = self.client.query_api()

    def fetch_data(self, measurement="EM-300", range_start="0"):
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {range_start})
          |> filter(fn: (r) => r["_measurement"] == "{measurement}")
          |> filter(fn: (r) => r["_field"] == "hum" or r["_field"] == "temp")
        '''
        
        try:
            tables = self.client.query_api().query(org=self.org, query=query)
            
            records_list = []
            for table in tables:
                for record in table.records:
                    records_list.append({
                        "_time": record.get_time(),
                        "_field": record.get_field(),
                        "_value": record.get_value()
                    })
            
            if not records_list:
                return pd.DataFrame()

            df_raw = pd.DataFrame(records_list)
            
            df_pivot = df_raw.pivot(index='_time', columns='_field', values='_value').reset_index()
            
            df_pivot = df_pivot.sort_values('_time').reset_index(drop=True)
            
            return df_pivot

        except Exception as e:
            print(f"❌ Error during manual fetch: {e}")
            return pd.DataFrame()
        
    def close(self):
        self.client.close()

