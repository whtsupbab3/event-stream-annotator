import collections

import numpy as np

def read_event_file(filename):
    packet_size = 8
    with open(filename, 'rb') as f:
        data = f.read()
    data = np.frombuffer(data, np.uint8)
    if len(data)%packet_size != 0:
        data = data[:-(len(data)%packet_size)]
    y = ((data[1::packet_size].astype('uint16')<<8) + data[::packet_size]) >> 2
    x = ((data[3::packet_size].astype('uint16')<<8) + data[2::packet_size]) >> 1
    p = np.where((data[::packet_size] & 0x02) == 0x02, 1, 0)
    # find the time stamp for each event, in microseconds from the start of the file
    t = data[7::packet_size].astype(np.uint32)
    t = (t << 8) + data[6::packet_size]
    t = (t << 8) + data[5::packet_size]
    t = (t << 8) + data[4::packet_size]
    
    t_diff = np.diff(t)
    wrap_indices = np.where(t_diff.view('int32')<0)[0]
    for index in wrap_indices:
        t[index+1:] += (2**31)
    
    
    t = t - t[0]
    t = t.astype('int32')
    x = x.astype('uint16')
    y = y.astype('uint16')
    p = p.astype('bool')
    Data = collections.namedtuple('Data', ['t', 'x', 'y', 'p'])
    return Data(t, x, y, p)
    
    

def read_h5_file(filename):
    # WACV2024 ICTrap .h5: CD/events is a compound array already decoded
    # to (x, y, p, t) fields, t in microseconds. No faery decoding needed.
    import h5py
    with h5py.File(filename, 'r') as f:
        events = f['CD/events'][:]
    Data = collections.namedtuple('Data', ['t', 'x', 'y', 'p'])
    t = events['t']
    return Data((t - t[0]).astype('int32'), events['x'].astype('uint16'),
                events['y'].astype('uint16'), events['p'].astype('bool'))


def read_raw_file(filename):
    import faery
    # If we want to read CSV
    # from faery.file_decoder import CsvProperties
    # csv_properties = CsvProperties(
    #     has_header=False,
    #     separator=b',',
    #     x_index=0,
    #     y_index=1,
    #     on_index=2,
    #     t_index=3,
    #     t_scale=0.0,
    #     on_value=b'1',
    #     off_value=b'0',
    #     skip_errors=False,
    # )
    # events = faery.events_stream_from_file(filename, csv_properties=csv_properties).to_array()
    events = faery.events_stream_from_file(filename).to_array()
    print(events.dtype)
    print(np.amax(events["x"]), np.amax(events["y"]))
    Data = collections.namedtuple('Data', ['t', 'x', 'y', 'p'])
    t = events["t"]
    return Data((t-t[0]).astype('int32'), events["x"].astype('uint16'), 
                events["y"].astype('uint16'), events["p"].astype('bool'))

    
def read(filename):
    if filename.endswith('.h5'):
        return read_h5_file(filename)
    return read_raw_file(filename)
        


# Lower values can also mask real moving objects in short (3-4 s) clips.
HOT_PIXEL_FRAC = 0.99


def drop_hot_pixels(data, min_active=HOT_PIXEL_FRAC, bin_us=1_000_000):
    # Drops pixels that fire in >= min_active of all 1 s bins.
    t, x, y = data.t.astype(np.int64), data.x.astype(np.int64), data.y.astype(np.int64)
    n_px = int(x.max() + 1) * int(y.max() + 1)
    idx = y * int(x.max() + 1) + x
    b = (t - t[0]) // bin_us
    n_bins = int(b.max()) + 1
    active = np.bincount(np.unique(idx * n_bins + b) // n_bins, minlength=n_px)
    hot = active >= min_active * n_bins
    keep = ~hot[idx]
    print(f'masked {hot.sum()} hot pixels ({100 * (1 - keep.mean()):.0f}% of events)')
    return type(data)(*(a[keep] for a in data))
