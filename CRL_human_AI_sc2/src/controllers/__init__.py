REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .non_shared_controller_hyper import NonSharedMACHyper

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["non_shared_mac_hyper"] = NonSharedMACHyper