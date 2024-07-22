import os

import logging
import pandas as pd
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Logger hanndlers for warning
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.WARNING)

c_format = logging.Formatter("\n%(asctime)s - %(name)s - %(levelname)s - %(message)s\n")
c_handler.setFormatter(c_format)

logger.addHandler(c_handler)

project_root = os.path.dirname(os.path.abspath(__file__))


class LoadData(ABC):
    def __init__(self, file_name, delimiter, alg_name, preprocess) -> None:
        self.file_name = file_name
        self._delimiter = delimiter
        self._alg_name = alg_name
        self.preprocess = preprocess

    @abstractmethod
    def open_file(self, parameters):
        """
        Overwritten by instances for file types
        """
        pass

    def preprocess_data(func):
        def wrapper(self, args):
            if self.preprocess:
                raise NotImplementedError
            else:
                return func(args)

        return wrapper

    @preprocess_data
    @abstractmethod
    def extract_data(self, parameters):
        """
        Overwritten for specific file type
        """
        pass

    @abstractmethod
    def save_data(self, parameters):
        """
        Overwritten for specific file type
        """

    @property
    def file_name(self):
        return self._file_name

    @file_name.setter
    def file_name(self, value):
        f_type = value.split(".")
        f_type = f_type[-1]
        type_lst = ["txt", "csv", "json"]
        try:
            assert f_type in type_lst
            self._file_name = value
        except AssertionError:
            logger.exception(
                f"For {self.file_name}:\n This file type is not supported. Ensure to use one of: {type_lst}"
            )
            raise


class TextCsvData(LoadData):
    def __init__(
        self,
        file_name: str,
        preprocess: bool,
        pre_func: object,
        alg_name: str,
        colnames: list = None,
        delimiter=None,
        skip=False,
    ):
        super().__init__(file_name, delimiter, alg_name, preprocess)
        self.colnames = colnames
        self.pre_func = pre_func
        self.skiplines = skip
        self.df_txt = self.open_file()
        self.df_ext = self.extract_data()

    def open_file(self):
        """
        Open file with all columns providing headers if needed and correct delimiters
        """
        try:
            if self.colnames:
                df_txt = pd.read_csv(
                    os.path.join(args.data_path, self._alg_name, self._file_name),
                    delimiter=self._delimiter,
                    usecols=range(len(self.colnames) + 1),
                    header=None,
                    skip_blank_lines=self.skiplines,
                    index_col=0,
                )
                df_txt.columns = self.colnames
            else:
                df_txt = pd.read_csv(
                    os.path.join(args.data_path, self._alg_name, self._file_name),
                    delimiter=self._delimiter,
                    skip_blank_lines=self.skiplines,
                    index_col=0,
                )

            return df_txt

        except ValueError:
            logger.exception(
                f"For {self.file_name}:\n Please ensure to provide colnames as input to Data."
            )
            raise

        except FileNotFoundError:
            logger.exception(
                f"For {self.file_name}:\n Ensure to input correct file name and to place within data/algname directory.\n It is likely that the config.json file is not correct for the file."
            )
            raise

    def preprocess_data(func):
        def wrapper(self):
            if self.preprocess:
                print(f"Preprocessing data for {self._alg_name}")
                self.pre_func(self)
                return func(self)
            else:
                return func(self)

        return wrapper

    @preprocess_data
    def extract_data(self):
        try:
            df_ext = self.df_txt[["target", "reactants"]].copy()
            return df_ext
        except KeyError:
            header_name = ["target", "reactants"]
            logger.exception(
                f"For {self.file_name}:\n The headers for target and reactants should follow the convention of:{header_name}."
            )

        except Exception:
            logger.exception(
                f"For {self.file_name}:\n Something went wrong while parsing. Please check data file for inconsistencies."
            )

    def save_data(self):
        try:
            self.df_ext.to_csv(
                os.path.join(
                    args.data_path, self._alg_name, self._alg_name + "_processed.csv"
                )
            )
            print(
                f"{self._alg_name} data saved to {args.data_path}/{self._alg_name} directory."
            )
        except:
            logger.warning(f"For {self.file_name}:\n Processed data unable to save.")


if __name__ == "__main__":

    from src.utils.data_preprocess import megan_preprocess
    import json
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Preprocess data for retrosynthesis algorithms")
    parser.add_argument(
        "--config_name", type=str, help="Name of config file to use", required=True
    )
    parser.add_argument(
        "--data_path", type=str, help="Location of data files", default="data"
    )
    parser.add_argument(
        "--config_path", type=str, help="Location of config files", default="config"
    )
    args = parser.parse_args()

    pre_funcs = {"megan": megan_preprocess}

    with open(os.path.join(args.config_path, args.config_name)) as f:
        configs = json.load(f)
    for name in configs.keys():
        alg_data = configs[name]
        file_type = alg_data["file"].split(".")[-1]
        types = ["txt", "csv", "json"]

        if file_type in types:
            delimiter = "," if alg_data["delimiter"] == "comma" else " "
            if alg_data["preprocess"]:
                try:
                    pre_func = pre_funcs[name]
                except KeyError:
                    logger.exception(
                        "Add the preprocess function to the pre_funcs dict."
                    )
            else:
                pre_func = None

            alg_data = TextCsvData(
                file_name=alg_data["file"],
                preprocess=alg_data["preprocess"],
                pre_func=pre_func,
                alg_name=name,
                colnames=alg_data["colnames"],
                delimiter=delimiter,
                skip=alg_data["skip"],
            )
            alg_data.save_data()

        else:
            logger.exception(
                f"File type {file_type} is not supported. Please use one of: {types}"
            )
