import json
import pandas as pd
import datetime

class ConvertToJSON:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def processValues(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime.date, datetime.datetime)):
            return obj.strftime('%Y-%m-%d')
        return obj

    def convert(self, output_file):
        print("Converting DataFrame to JSON")
        json_data = self.dataframe.map(self.processValues).to_dict(orient='records')

        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=4)
        
        print(f"JSON data has been written to {output_file}")
