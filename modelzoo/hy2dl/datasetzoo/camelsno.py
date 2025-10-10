# import necessary packages
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from hy2dl.datasetzoo.basedataset import BaseDataset


class CAMELS_NO(BaseDataset):
    """Class to process data from the CAMELS US dataset [1]_ [2]_.

    The class inherits from BaseDataset to execute the operations on how to load and process the data. However here we
    code the _read_attributes and _read_data methods, that specify how we should read the information from CAMELS-US.

    This class and its methods were adapted from Neural Hydrology [3]_.

    Parameters
    ----------
    dynamic_input : Union[List[str], Dict[str, List[str]]]
        Name of variables used as dynamic series input in the data driven model. In most cases is a single list. If we
        are using multiple frequencies, it is a dictionary where the key is the frequency and the value is a list of
        variables.
    forcing : List[str]
        specificy which forcing data will be used (e.g. daymet, maurer, ndlas, ndlas_hourly, etc.)
    target : List[str]
        Target variable that will be used to train the model
    sequence_length : int
        Sequence length used for the model
    time_period : List[str]
        Initial and final date of the time period of interest. It should be consistent with the resolution of the data.
        e.g.  daily: ["1990-10-01","2003-09-30"], hourly: ["1990-10-01 00:00:00","2003-09-30 23:00:00"]
    path_data : str
        Path to the folder were the data is stored
    path_entities : str = None
        Path to a txt file that contain the id of the entities (e.g. catchment`s ids) that will be analyzed. An
        alternative option is to specify the entity directly with the "entity" parameter.
    entity : str = None
        id of the entity (e.g. single catchment`s id) that will be analyzed. An alternative option is to specify the
        path to a txt file containing multiple entities, using the "path_entities" parameter.
    check_NaN : Optional[bool] = True
        Boolean that indicate if one should check of NaN values while processing the data. This parameter should be
        True during training, and it can be switch to False during evaluation (validation/testing)
    path_addional_features : Optional[str] = None
        Allows the option to add any arbitrary data that is not included in the standard data sets. Path to a pickle
        file, containing a dictionary with each key corresponding to one basin id  and the value is a date-time indexed
        pandas DataFrame, where the columns are the additional features. Default value is None.
        For deliberate exclusion of samples, the dictionary's value is a date-time indexed pandas DataFrame with one
        single column named "ablation_flag", containing 0/1 flags (0 for exclusion).
    predict_last_n : Optional[int] = 1
        Number of timesteps (e.g. days, hours) that will be output by the model as predictions. Default value is 1.
    static_input : Optional[List[str]] = None
        Name of static attributes used as input in the lstm (e.g. catchment attributes). Default value is None.
    conceptual_input : Optional[List[str]] = None
        Name of variables used as dynamic series input in the conceptual model. Mandatory when using hybrid models.
    custom_freq_processing : Optional[Dict[str, int]] = None
        Necessary when using multiple frequencies (e.g. daily/hourly). We specify the information as a nested
        dictionary. The key of the first dictionary is the frequency name (e.g "1D", "1h"...). The values of the
        first dictionary (keys of the second one) are "freq_factor" and "n_steps". The values of the second dictionary
        are the respective values. The order of the frequencies should be consistent with the way we want to process
        the data (e.g. first daily, then hourly). In case we have more than two frequencies, the frequency factor is
        always relative to the higher frequency. Default value is None.
        e.g. custom_freq_processing = {"1D": {"freq_factor": 24, "n_steps": 351}, "1h": {"freq_factor": 1, "n_steps": 14*24}}
    dynamic_embedding : Optional[bool] = False
        True if we want to use embeddings (custom linear layers) for the different frequencies. This is necessary when
        one has different number of inputs for each frequency.
    unique_prediction_blocks : Optional[bool] = False
        If predicting_last_n>1, we can decide to use unique_prediction_blocks = True if we do not want to have
        overlapping blocks of training data.

    References
    ----------
    .. [1] A. J. Newman, M. P. Clark, K. Sampson, A. Wood, L. E. Hay, A. Bock, R. J. Viger, D. Blodgett,
        L. Brekke, J. R. Arnold, T. Hopson, and Q. Duan: Development of a large-sample watershed-scale
        hydrometeorological dataset for the contiguous USA: dataset characteristics and assessment of regional
        variability in hydrologic model performance. Hydrol. Earth Syst. Sci., 19, 209-223,
        doi:10.5194/hess-19-209-2015, 2015
    .. [2] Addor, N., Newman, A. J., Mizukami, N. and Clark, M. P.: The CAMELS data set: catchment attributes and
        meteorology for large-sample studies, Hydrol. Earth Syst. Sci., 21, 5293-5313, doi:10.5194/hess-21-5293-2017,
        2017.
    .. [3] F. Kratzert, M. Gauch, G. Nearing and D. Klotz: NeuralHydrology -- A Python library for Deep Learning
        research in hydrology. Journal of Open Source Software, 7, 4050, doi: 10.21105/joss.04050, 2022

    """

    def __init__(
        self,
        dynamic_input: Union[List[str], Dict[str, List[str]]],
        forcing: List[str],
        target: List[str],
        sequence_length: int,
        time_period: List[str],
        path_data: str,
        path_entities: str = None,
        entity: str = None,
        check_NaN: bool = True,
        path_additional_features: Optional[str] = "",
        predict_last_n: Optional[int] = 1,
        static_input: Optional[List[str]] = None,
        conceptual_input: Optional[List[str]] = None,
        custom_freq_processing: Optional[Dict[str, int]] = None,
        dynamic_embedding: Optional[bool] = False,
        unique_prediction_blocks: Optional[bool] = False,
    ):
        # Run the __init__ method of BaseDataset class, where the data is processed
        self.forcing = forcing
        super(CAMELS_NO, self).__init__(
            dynamic_input=dynamic_input,
            target=target,
            sequence_length=sequence_length,
            time_period=time_period,
            path_data=path_data,
            path_entities=path_entities,
            entity=entity,
            check_NaN=check_NaN,
            path_additional_features=path_additional_features,
            predict_last_n=predict_last_n,
            static_input=static_input,
            conceptual_input=conceptual_input,
            custom_freq_processing=custom_freq_processing,
            dynamic_embedding=dynamic_embedding,
            unique_prediction_blocks=unique_prediction_blocks,
        )

    def _read_data(self, catch_id: str) -> pd.DataFrame:
        nc_path = Path(self.path_data) / "time_series" / f"{catch_id}.nc"
        ds = xr.open_dataset(nc_path)
        df = ds.to_dataframe().reset_index().set_index("date")
        df.rename(columns={"streamflow": "QObs(mm/h)"}, inplace=True)
        df["QObs(mm/h)"] = df["QObs(mm/h)"].apply(lambda x: np.nan if x < 0 else x)
        return df
    

    def _read_attributes(self) -> pd.DataFrame:
        attr_path = Path(self.path_data) / "attributes" / "attributes.csv"
        df = pd.read_csv(attr_path).set_index("basin_id")
        df = df.loc[self.entities_ids]
        df = df[self.static_input]  # filter to only static_input variables
        return df

