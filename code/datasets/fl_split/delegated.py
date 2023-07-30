import logging
from itertools import repeat
from typing import final, Sequence, Optional
from code.datasets.raw_dataset.base import _RawDataset
from code.datasets.fl_split.base import _HorizontalFLSplit
from munch import Munch

from code.datasets.utils import load_obj, metadata_updated, save_dict, save_obj

log = logging.getLogger(__name__)


@final
class DelegatedSplit(_HorizontalFLSplit):
    """
    Delegate the dataset splitting process to the single underlying dataset.
    Similar to _RealisticSplit but requires only one underlying raw dataset.
    Used when post-processing after splitting a single dataset is needed.
    Allows more flexible data splitting

    When implmeenting a raw dataset class intended to be used with this class,
    the raw dataset class should process num_splits and split_id and return
    different descriptor lists for each party when load_*_descs is called.
    """

    def __init__(
            self, cfg: Munch, datasets: Sequence[_RawDataset],
            mode, split_id: Optional[int]):

        assert len(datasets) == 1, \
            "DeletagedSplit no more than one underlying raw dataset"

        super().__init__(cfg, datasets, mode, split_id)
        self.split_id = split_id
        self.preproces_and_cache()

    def _load_desc(self):
        method = f"load_{self.mode}_descs"
        return list(zip(
            getattr(self.datasets[0], method)(), repeat(0)))

    def _preprocess_train_descs(self, descs):
        return descs
