import pandas as pd
from dataclasses import dataclass


@dataclass
class GetData:
    """
    Get data from data source. Currently this is only used to read csv files so we dont really need a class but this allows
    us to consider the future and add other data types and/or checks for data quality.
    """

    path: str

    def prepare(self):
        """
        Read data from csv file
        """
        df = pd.read_csv(self.path)
        df.columns = ["".join(x.lower().split()) for x in df.columns]
        return df
