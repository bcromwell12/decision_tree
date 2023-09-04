from .steps import GetData, FeatureBuilding

data = GetData(path="data/animals.csv").preapre()
x_data, y_data = FeatureBuilding(data).normal_index_encoding()