REGISTRY = {}

from .non_shared_controller import NonSharedMAC
from .non_shared_controller_hyper import NonSharedMACHyper

REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["non_shared_mac_hyper"] = NonSharedMACHyper