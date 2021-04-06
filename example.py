import os
import pyxdf
from xdf2mne import stream2raw

import logging
logger = logging.getLogger(__name__)

filename = 'your_own.xdf'
filepath = os.path.join('./', filename)

streams, fileheader = pyxdf.load_xdf(filepath, dejitter_timestamps=True)

##%
stream = streams[0]
marker_stream = streams[1]

raw, events, event_id = stream2raw(stream, marker_stream=marker_stream)
