"""
This module provides methods to transform lsl streams as loaded from a .xdf-file with pyxdf into mne.io.RawArray objects
mne itself does not provide such method
The methods I implemented here are not very sophisticated and do not check many edge cases. The module assumes that
the timestamps for data and marker streams are synchronized and that the stream's sampling rate matches the
nominal sampling rate exactly. Usage of pyxdf's dejitter_timestamps flag when loading the streams is recommended.
If you recorded the data from different devices with unsynced clocks, the timestamps need to be fixed before using this.
"""
import numpy as np
import mne

import logging
logger = logging.getLogger(__name__)


def ch_type_transform_default(type_):
    type_ = str(type_).lower()
    if type_ in mne.io.pick.get_channel_type_constants():
        return type_
    else:
        return 'misc'


def marker_stream2events(marker_stream, t_reference, sfreq):
    """
    Extract the event markers from a marker stream and create an events array as well as an event_id dict
    Since mne expects indices instead of timestamps, this method requires an array t_reference in relation
    to which indices are determined. Expects the marker_stream and t_reference timestamps to be from synced clocks.
    The event_ids are extracted from channel 0 of marker_stream and uniquely enumerated.
    event_id is a dictionary mapping the event_ids to the enumeration
    :param marker_stream: A marker stream from an xdf-file
    :param t_reference: An array of timestamps which is used to determine timestamp indices of the events
    :return: events and event_id compatible with mne
    """

    # Extract channel 0 from the marker stream since mne can't handle more
    markers = np.array(marker_stream['time_series'])[:, 0]
    markers_t = marker_stream['time_stamps']

    # Shift t_reference by half a sampling period to adjust bin centers.
    # Digitize the markers into the bins and resolve the indices
    markers_ref_idx = np.digitize(markers_t, bins=t_reference+((1 / sfreq) / 2))

    event_id = {name: i for i, name in enumerate(np.unique(markers))}
    marker_ids = list(map(event_id.get, markers))
    events = np.array([markers_ref_idx, np.zeros(len(marker_ids)), marker_ids], dtype=int).T

    return events, event_id


def stream2raw(stream, marker_stream=None, ch_type_transform=ch_type_transform_default):
    """
    Transform a data stream and an optional marker stream into an mne.io.RawArray object
    Takes ch_type_transform to transform type declarations between xdf and mne:
        If not given, uses 'misc' if channel type is invalid for mne
        If given, should be function with single parameter to transform the type
        e.g. lambda t: 'meg' if t == '1' else 'misc'
    For handling of markers see marker_stream2events
    return:
    - raw: RawArray object
    - events: None if no marker stream given, otherwise numpy array with shape (3, n_events) as expected by mne
    - event_id: dictionary that maps the marker values (from marker_stream channel 0) to events
    """
    # Extract information from stream
    t_original = stream['time_stamps']
    sfreq = np.float(stream['info']['nominal_srate'][0])
    stream_type = ch_type_transform(stream['info']['type'][0])
    data = stream['time_series'].T  # Transpose to make shape (n_channels, n_times)
    channel_count = int(stream['info']['channel_count'][0])

    desc = stream['info']['desc'][0]

    ch_names = []
    ch_types = []
    if 'channels' in desc:
        for channel in desc['channels'][0]['channel']:
            ch_names.append(channel['label'][0])
            if 'type' in channel:
                ch_types.append(ch_type_transform(channel['type'][0]))
            else:
                ch_types.append(stream_type)


    if len(ch_names) != channel_count:
        logger.warning(f"Description is missing {channel_count-len(ch_names)} channel names. "
                       f"Resorting to enumeration")
        ch_names = channel_count  # mne's `create_info` takes an int for enumeration

    elif len(ch_names) != len(ch_types):
        logger.warning(f"Channel types not consistently given. Resorting to stream channel type")
        ch_types = stream_type  # mne's `create_info` also takes a string

    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(data=data, info=info)
    raw.info['t_original'] = t_original


    events, event_id = None, None
    if marker_stream:
        events, event_id = marker_stream2events(marker_stream, t_original, sfreq)

    return raw, events, event_id
