"""
This module provides methods to transform lsl streams as loaded from a .xdf-file with pyxdf into mne.io.RawArray objects
mne itself does not provide such method
The methods I implemented here are not very sophisticated and do not check many edge cases. The module assumes that
the timestamps for data and marker streams are synchronized and that the stream's sampling rate matches the
nominal sampling rate exactly. Usage of pyxdf's dejitter_timestamps flag when loading the streams is recommended.
If you recorded the data from different devices with unsynced clocks, the timestamps need to be fixed before using this.
"""
from typing import Union, List, Callable

import numpy as np
import mne
import pyxdf

import utils

import logging
logger = logging.getLogger(__name__)

CH_TYPE_DEFAULT = 'misc'

def ch_type_transform_default(type_=None, default_type=CH_TYPE_DEFAULT):
    if default_type is None or default_type not in mne.io.pick.get_channel_type_constants():
        default_type = CH_TYPE_DEFAULT

    if type_ is None:
        return default_type

    type_ = str(type_).lower()
    return type_ if type_ in mne.io.pick.get_channel_type_constants() else default_type

def _get_event_id(marker_stream):
    markers = np.array(marker_stream['time_series'])[:, 0]
    event_id = {name: i for i, name in enumerate(np.unique(markers))}
    return event_id

def marker_stream2annotations(marker_stream, t_reference, t_duration=0.0):
    """
    Extract the events from a marker stream and create an mne.Annotations object
    This method requires an array or integer t_reference in relation to which "t=0" is determined.
    Expects the marker_stream and t_reference timestamp(s) to be from synced clocks.
    If there are multiple channels in marker_stream, they are concatenated with using '/'
    This allows use of epoch subselectors in mne
    https://mne.tools/dev/auto_tutorials/epochs/10_epochs_overview.html#tut-section-subselect-epochs
    :param marker_stream: A marker stream from an xdf-file
    :param t_reference: A single timestamp or an array of timestamps determine t=0
    :param t_duration: Default duration of the annotations. Could be extended in the future
    :return: mne.Annotations object containing all markers, event_id dictionary compatible with mne
    """
    # Extract channel 0 from the marker stream since mne can't handle more
    # markers = np.array(marker_stream['time_series'])[:, 0]
    markers = np.array(marker_stream['time_series']).astype(str)
    marker_strings = ['/'.join(markers[i]) for i in range(len(markers))]
    markers_t = marker_stream['time_stamps']-np.min(t_reference)

    # event_id = _get_event_id(marker_stream)
    annotations = mne.Annotations(onset=markers_t, duration=[t_duration] * len(markers_t), description=marker_strings)
    return annotations


def marker_stream2events(marker_stream, t_reference, sfreq):
    """
    Extract the event markers from a marker stream and create an events array as well as an event_id dict
    Since mne expects indices instead of timestamps, this method requires an array t_reference in relation
    to which indices are determined. Expects the marker_stream and t_reference timestamps to be from synced clocks.
    The event_ids are extracted from channel 0 of marker_stream and uniquely enumerated.
    event_id is a dictionary mapping the event_ids to the enumeration
    :param marker_stream: A marker stream from an xdf-file
    :param t_reference: An array of timestamps which is used to determine timestamp indices of the events
    :param sfreq: The sampling frequency of the data

    :return: events and event_id compatible with mne
    """

    # Extract channel 0 from the marker stream since mne can't handle more
    markers = np.array(marker_stream['time_series'])[:, 0]
    markers_t = marker_stream['time_stamps']

    # Shift t_reference by half a sampling period to adjust bin centers.
    # Digitize the markers into the bins and resolve the indices
    markers_ref_idx = np.digitize(markers_t, bins=t_reference+((1 / sfreq) / 2))

    event_id = _get_event_id(marker_stream)
    marker_ids = list(map(event_id.get, markers))
    events = np.array([markers_ref_idx, np.zeros(len(marker_ids)), marker_ids], dtype=int).T

    return events, event_id


# def stream2raw(stream, ch_type_t=ch_type_transform_default):
#     t_original = stream['time_stamps']
#     sfreq_nominal = np.float(stream['info']['nominal_srate'][0])
#     stream_type = ch_type_t(stream['info']['type'])
#     data = stream['time_series'].T  # Transpose for shape (n_channels, n_times)
#     n_channels, n_samples = data.shape
#
#     desc = stream['info']['desc'][0]

def streams2raw(data_stream: dict, marker_streams: List[dict] = None, ch_type_t: Callable = ch_type_transform_default) -> mne.io.RawArray:
    raw = stream2raw(data_stream, ch_type_t=ch_type_t)
    raw_add_annotations(raw, marker_streams=marker_streams)
    return raw



def stream2raw(stream, ch_type_t=None) -> mne.io.RawArray:
    """
    Transform a data stream and an optional marker stream into an mne.io.RawArray object
    Takes a function ch_type_transform(x) to transform type declarations between xdf and mne:
        If not given, uses 'misc' if channel type is invalid for mne
        If given, should be function with single parameter to transform the type
        e.g. lambda _type: 'meg' if _type == '1' else 'misc'
    For handling of markers see marker_stream2events
    If a marker stream is given, annotations are automatically generated and attached to raw
    return:
    - raw: RawArray object
    - events: None if no marker stream given, otherwise numpy array with shape (3, n_events) as expected by mne
    - event_id: dictionary that maps the marker values (from marker_stream channel 0) to events
    """


    # Extract information from stream
    t_original = stream['time_stamps']
    sfreq = np.float(stream['info']['nominal_srate'][0])

    if sfreq <= 0:
        n_samples = len(stream['time_stamps'])
        effective_srate = n_samples/(np.ptp(stream['time_stamps']))
        logger.warning(f"nominal_srate must be positive. Using {effective_srate=} Hz instead")
        sfreq = effective_srate

    if ch_type_t is None:
        def ch_type_transform(type_=None):
            return ch_type_transform_default(type_, stream['info']['type'][0])
        ch_type_t = ch_type_transform

    stream_type = ch_type_t(stream['info']['type'][0])
    data = stream['time_series'].T  # Transpose to make shape (n_channels, n_times)

    ch_names, ch_types = get_ch_info(stream['info'], ch_type_t=ch_type_t)

    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(data=data, info=info)
    raw._t_original = t_original

    return raw


def raw_add_annotations(raw: mne.io.RawArray, marker_streams=None) -> mne.io.RawArray:
    t_reference = raw._t_original
    if marker_streams:
        for stream in marker_streams:
            annotations = marker_stream2annotations(stream, t_reference=t_reference)
            raw.set_annotations(raw.annotations + annotations)

    return raw


def get_ch_info(info, ch_type_t=ch_type_transform_default):
    ch_names = []
    ch_types = []

    channel_count = int(info['channel_count'][0])

    if 'desc' in info:
        desc = info['desc'][0]
        if 'channels' in desc and 'channel' in desc['channels'][0]:
            for ch_info_idx, channel in enumerate(desc['channels'][0]['channel']):
                if 'label' in channel:
                    ch_label = channel['label'][0]
                    ch_names.append(ch_label)

                    if 'type' in channel:
                        ch_types.append(ch_type_t(channel['type'][0]))
                    else:
                        logger.warning(f"Channel {ch_label} is missing a type")
                        ch_types.append(ch_type_t())
                else:
                    break
                    # If a channel has no label, the "channels" structure seems corrupted and we use fallback to defaults

    if len(ch_names) != channel_count:
        logger.warning(f"Info is missing information on {channel_count - len(ch_names)} channels. "
                       f"Resorting to enumeration of channels and default type {ch_type_t()}")
        ch_names = [f'channel{i:0{np.ceil(np.log10(channel_count)).astype(int)}}' for i in range(channel_count)]
        ch_types = [ch_type_t()]*channel_count

    return ch_names, ch_types


def streams2dict(streams_list, header=None, return_header=False):
    """
    Can be called with a list of streams or by calling streams2dict(*pyxdf.load_xdf(filename, ...))
    """
    streams = {stream['info']['name'][0]: stream for stream in streams_list}
    if return_header:
        return streams, header
    else:
        return streams
