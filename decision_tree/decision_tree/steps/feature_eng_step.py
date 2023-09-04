import pandas as pd


class FeatureBuilding:
    """
    Class for feature engineering. We want a class for this because we want to be able to add new techniques over time

    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.color_encoder = dict()
        self.leg_encoder = dict()
        self.name_encoder = dict()

    @staticmethod
    def index_data(df: pd.Series) -> (pd.Series, dict):
        """
        Index data
        """
        d = {k: v for v, k in enumerate(df.unique(), 1)}
        dat = df.map(d)
        return dat, d

    @staticmethod
    def one_hot_encoding(df: pd.Series):
        """
        would do one hot encoding.
        :param df:
        :return:
        """
        pass

    @staticmethod
    def word_embedding(df: pd.Series, model):
        """
        would do word embedding. based on the model passed
        :param df:
        :param model: embeding model to be used
        :return:
        """
        pass

    def normal_index_encoding(self, label_name: str = "name") -> pd.DataFrame:
        """
        Feature engineering. Here we would allow for multiple types of feature buildinig so we can test different tehcniques.
        """
        x_data = self.data.loc[:, self.data.columns != label_name]
        y_data = self.data.loc[:, label_name]
        x_data["encoded_color"], self.color_encoder = self.index_data(x_data["color"])
        x_data["encoded_legs"], self.name_encoder = self.index_data(x_data["numberoflegs"])
        y_data['encoded_names'], self.name_encoder = self.index_data(y_data)
        # ideally here we would build out the names to all have encoders then we could sort them by the existance of encoder
        # when it comes to the the full production so our data can change but for now i will write them out for speed

        return x_data, y_data


