# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.core.config import cfg
from pysot.tracker.siamrpn_tracker import SiamRPNTracker
from pysot.tracker.siammask_tracker import SiamMaskTracker
from pysot.tracker.siamrpnlt_tracker import SiamRPNLTTracker
from pysot.tracker.siamfc_tracker import SiamFCTracker
from pysot.tracker.DAsiamrpn_tracker import DASiamRPNTracker
from pysot.tracker.DAsiamrpnlt_tracker import DASiamRPNLTTracker
from pysot.tracker.ocean_tracker import OceanTracker
from pysot.tracker.oceanonline_tracker import OceanOnlineTracker
from pysot.tracker.siamcar_tracker import SiamCARTracker



TRACKS = {
    'SiamRPNTracker': SiamRPNTracker,
    'SiamMaskTracker': SiamMaskTracker,
    'SiamRPNLTTracker': SiamRPNLTTracker,
    'SiamFCTracker': SiamFCTracker,
    'DASiamRPNTracker': DASiamRPNTracker,
    'DASiamRPNLTTracker': DASiamRPNLTTracker,
   # 'SiamRPNBANTracker': SiamRPNBANTracker,
    'OceanTracker': OceanTracker,
     'OceanOnlineTracker': OceanTracker,
   'SiamCARTracker' :SiamCARTracker,
}


TRACKS_ONLINE = {
    'OceanOnlineTracker': OceanOnlineTracker,
}


def build_tracker(model, dataset=None):
    return TRACKS[cfg.TRACK.TYPE](model, dataset)


def build_tracker_online(model, dataset=None, checkpoint_name='', online= False):
    return TRACKS[cfg.TRACK.TYPE](model, dataset, checkpoint_name, online), TRACKS_ONLINE[cfg.TRACK.TYPE](model, dataset, checkpoint_name, online)


# def build_tracker_online(model, dataset=None):
#     return  None, TRACKS_ONLINE[cfg.TRACK.TYPE](model, dataset)





# def build_tracker_online(model, dataset=None):
#     return TRACKS_ONLINE[cfg.TRACK.TYPE](model, dataset), TRACKS_ONLINE[cfg.TRACK.TYPE](model, dataset)
