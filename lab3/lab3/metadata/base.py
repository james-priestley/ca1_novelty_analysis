import pandas as pd
import os
from lab3.core.metadata import Metadata, MetadataGroup

class FileMetadata(Metadata):
    constructor_ids = ['file_index_common', 'df_index_common', 'metadata_columns', 'file_name']

    def bind(self, expt):
        return pd.MultiIndex.from_frame(
            self.read_file_data(expt)[self.metadata_columns])

    def bind_to(self, signals, expt):
        filedata = self.read_file_data(expt)
        filedata[self.df_index_common] = filedata[self.file_index_common]
        filedata = filedata.set_index(self.df_index_common)

        signals_index = signals.reset_index()[self.df_index_common].iloc[0]
        
        try:
            signals_index.shape
            index_type = [type(i) for i in signals_index]
        except:
            index_type = type(signals_index)

        filedata.index = filedata.index.astype(index_type)

        #dtype = signals.reset_index()[self.df_index_common].dtype

        return signals.join(filedata[self.metadata_columns],
                            on=[self.df_index_common]).set_index(
                            self.metadata_columns, append=True)

    def read_file_data(self, expt):
        raise NotImplementedError

class ExcelMetadata(FileMetadata):
    def read_file_data(self, expt):
        filedata = pd.read_excel(os.path.join(expt.sima_path, self.file_name))
        return filedata

class CSVMetadata(FileMetadata):
    def read_file_data(self, expt):
        filedata = pd.read_csv(os.path.join(expt.sima_path, self.file_name))
        return filedata



