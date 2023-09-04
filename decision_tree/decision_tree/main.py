from decision_tree.steps import GetData, FeatureBuilding
from decision_tree.model import DecisionTree

data = GetData(path="data/animals.csv").prepare()
x_data, y_data = FeatureBuilding(data).normal_index_encoding()
#forgot to only include columns with encode here. Not fully sure if i would do it inside feature builder or here in the main

model = DecisionTree(max_depth=10, min_size=1)
model.fit(x_data, y_data)

print(x_data)