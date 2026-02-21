from ._base import Vanilla
from .KD import VanillaKD
from .FitNet import FitNet
from .AT import AT
from .CC import CC
from .SP import SP
from .RKD import RKD
from .PKT import PKT
from .CSD import CSD
from .D3 import D3
from .PDRD import PDRD

distiller_dict = {
    "NONE": Vanilla,
    "VanillaKD": VanillaKD,
    "FitNet": FitNet,
    "AT": AT,
    "CC": CC,
    "SP": SP,
    "RKD": RKD,
    "PKT": PKT,
    "CSD": CSD,
    "D3": D3,
    "PDRD": PDRD
}
