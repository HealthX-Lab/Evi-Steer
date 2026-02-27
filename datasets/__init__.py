
from .btmri import BTMRI
from .btmri_p import BTMRI_P
from .btmri_s import BTMRI_S
from .brisc import BRISC

from .busi import BUSI
from .buid import BUID
from .busbra import BUSBRA
from .udiat import UDIAT

from .kvasir import Kvasir
from .ctkidney import CTKidney
from .chmnist import CHMNIST
from .lungcolon import LungColon
from .retina import RETINA
from .covid import COVID_19
from .octmnist import OCTMNIST

dataset_list = {
                "busi": BUSI,
                "buid": BUID,
                "busbra": BUSBRA,
                "udiat": UDIAT,
                "btmri": BTMRI,
                "brisc": BRISC,
                "btmri_p": BTMRI_P,
                "btmri_s": BTMRI_S,
                "ctkidney": CTKidney,
                "chmnist": CHMNIST,
                "kvasir": Kvasir,
                "lungcolon": LungColon,
                "retina": RETINA,
                "covid": COVID_19,
                "octmnist": OCTMNIST,
                }


def build_dataset(dataset, root_path, shots, template, subsample):
    return dataset_list[dataset](root_path, shots, template, subsample)