'''

Original file: export.py
Author: J. Binas <jbinas@gmail.com>, 2017

This software is released under the
GNU LESSER GENERAL PUBLIC LICENSE Version 3.

-------------------------------------------------------------------------------

Modified in order to compute several hdf5 files from a directory, and save
positive and negative event frames by separately.

'source_folder' and 'dest_folder' must be different to not overwrite the
original hdf5 files from DDD17. You could want to generate frame-based
data again but with a different time interval.

-------------------------------------------------------------------------------

Creates another hdf5 file containing frame-based data as a result of accumulating
the events every other binsize seconds.

'''


from __future__ import print_function
import os, time, argparse
import Queue
import numpy as np
import glob
from copy import deepcopy
from view import HDF5Stream, MergedStream
from datasets import HDF5
from interfaces.caer import DVS_SHAPE, unpack_data


export_data_vi = {
        'steering_wheel_angle',
        'brake_pedal_status',
        'accelerator_pedal_position',
        'engine_speed',
        'vehicle_speed',
        'windshield_wiper_status',
        'headlamp_status',
        'transmission_gear_position',
        'torque_at_transmission',
        'fuel_level',
        'high_beam_status',
        'ignition_status',
        #'lateral_acceleration',
        'latitude',
        'longitude',
        #'longitudinal_acceleration',
        'odometer',
        'parking_brake_status',
        #'fine_odometer_since_restart',
        'fuel_consumed_since_restart',
    }

export_data_dvs = {
        'dvs_frame',
        'aps_frame',
    }

export_data = export_data_vi.union(export_data_dvs)


def filter_frame(d):
    '''
    receives 8 bit frame,
    needs to return unsigned 8 bit img
    '''
    # add custom filters here...
    # d['data'] = my_filter(d['data'])
    frame8 = (d['data'] / 256).astype(np.uint8)
    return frame8

def get_progress_bar():
    try:
        from tqdm import tqdm
    except ImportError:
        print("\n\nNOTE: For an enhanced progress bar, try 'pip install tqdm'\n\n")
        class pbar():
            position=0
            def close(self): pass
            def update(self, increment):
                self.position += increment
                print('\r{}s done...'.format(self.position)),
        def tqdm(*args, **kwargs):
            return pbar()
    return tqdm(total=(tstop-tstart)/1e6, unit_scale=True)


def raster_evts(data):
    _histrange = [(0, v) for v in DVS_SHAPE]
    pol_on = data[:,3] == 1
    pol_off = pol_on == False
    # Positive-event frame
    img_on, _, _ = np.histogram2d(
            data[pol_on, 2], data[pol_on, 1],
            bins=DVS_SHAPE, range=_histrange)
    # Negative-event frame
    img_off, _, _ = np.histogram2d(
            data[pol_off, 2], data[pol_off, 1],
            bins=DVS_SHAPE, range=_histrange)
    img_on = np.expand_dims(img_on, axis=-1)
    img_off = np.expand_dims(img_off, axis=-1)
    # First channel of a DVS frame contains the positve events, and the second
    # one contains the negative events.
    dvs_img = np.concatenate((img_on, img_off), axis=-1)
    return dvs_img.astype(np.int16)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tstart', type=int, default=0, help='Starting time in seconds')
    parser.add_argument('--tstop', type=int, help='Stopping time in seconds')
    parser.add_argument('--binsize', type=float, default=0.1, help='If positive, events will be binned into time-slices'
                                                                   'of BINSIZE seconds. If negative, frames will be'
                                                                   'generated every abs(BINSIZE) events.')
    parser.add_argument('--update_prog_every', type=float, default=0.01, help='Progress bar update interval (in'
                                                                              'percentage points).')
    parser.add_argument('--export_aps', type=int, default=1, help='Whether to generate APS/grayscale frames')
    parser.add_argument('--export_dvs', type=int, default=1, help='Whether to generate DVS/event frames')
    parser.add_argument('--source_folder', help='Path to HDF5 files.')
    parser.add_argument('--dest_folder', help='Path to where saving the new HDF5 files.')
    args = parser.parse_args()

    recordings = glob.glob(args.source_folder + '/*')

    # For every recording/hdf5 file
    for rec in recordings:
            
        f_in = HDF5Stream(rec, export_data_vi.union({'dvs'}))
        m = MergedStream(f_in)

        fixed_dt = args.binsize > 0
        tstart = int(m.tmin + 1e6 * args.tstart)
        tstop = m.tmin + 1e6 * args.tstop if args.tstop is not None else m.tmax
        print(tstart, tstop)
        m.search(tstart)
    
        print('recording duration', (m.tmax - m.tmin) * 1e-6, 's')
    
        # Create output file
        dtypes = {k: float for k in export_data.union({'timestamp'})}
        if args.export_aps:
            dtypes['aps_frame'] = (np.uint8, DVS_SHAPE)
        if args.export_dvs:
            dtypes['dvs_frame'] = (np.int16, DVS_SHAPE+(2,))
    
        if not os.path.exists(args.dest_folder):
            os.makedirs(args.dest_folder)
        outfile = os.path.join(args.dest_folder, rec.split('/')[-1])
        f_out = HDF5(outfile, dtypes, mode='w', chunksize=32, compression='gzip')
    
        current_row = {k: 0 for k in dtypes}
        if args.export_aps:
            current_row['aps_frame'] = np.zeros(DVS_SHAPE, dtype=np.uint8)
        if args.export_dvs:
            current_row['dvs_frame'] = np.zeros(DVS_SHAPE+(2,), dtype=np.int16)
    
        pbar = get_progress_bar()
        sys_ts, t_pre, t_offset, ev_count, pbar_next = 0, 0, 0, 0, 0
        while m.has_data and sys_ts <= tstop*1e-6:
            try:
                sys_ts, d = m.get()
            except Queue.Empty:
                # Continue while waiting for queue to fill up
                continue
            if not d:
                # Skip unused data
                continue
            if d['etype'] == 'special_event':
                unpack_data(d)
                if any(d['data'] == 0):
                    d['etype'] = 'timestamp_reset'
                    current_row['timestamp'] = d['timestamp']
            if d['etype'] == 'timestamp_reset':
                print('ts reset detected, setting offset', current_row['timestamp'])
                t_offset += current_row['timestamp']
                continue
            if d['etype'] in export_data_vi:
                current_row[d['etype']] = d['data']
                continue
            if d['etype'] == 'frame_event' and args.export_aps:
                if t_pre == 0:
                    print('resetting t_pre (current frame)')
                    t_pre = d['timestamp'] + t_offset
                while fixed_dt and t_pre + args.binsize < d['timestamp'] + t_offset:
                    # SYSTEM Timestamp version:
                    current_row['timestamp'] = (sys_ts - tstart * 1e-6)
                    f_out.save(deepcopy(current_row))
                    current_row['dvs_frame'][:,:,:] = 0
                    current_row['timestamp'] = t_pre
                    t_pre += args.binsize
                if not fixed_dt:
                    current_row['timestamp'] = d['timestamp'] + t_offset
                current_row['aps_frame'] = filter_frame(unpack_data(d))
                current_row['timestamp'] = t_pre
                continue
            if d['etype'] == 'polarity_event' and args.export_dvs:
                unpack_data(d)
                times = d['data'][:,0] * 1e-6 + t_offset
                num_evts = d['data'].shape[0]
                if t_pre == 0:
                    print('resetting t_pre (current pol)')
                    t_pre = times[0]
                offset = 0
                if fixed_dt:
                    # fixed time interval bin mode
                    num_samples = np.ceil((times[-1] - t_pre) / args.binsize)
                    for _ in xrange(int(num_samples)):
                        # take n events
                        n = (times[offset:] < t_pre + args.binsize).sum()
                        sel = slice(offset, offset + n)
                        current_row['dvs_frame'] += raster_evts(d['data'][sel])
                        offset += n
                        # save if we're in the middle of a packet, otherwise
                        # wait for more data
                        if sel.stop < num_evts:
                            # SYSTEM Timestamp version:
                            current_row['timestamp'] = (sys_ts - tstart * 1e-6)
                            #current_row['timestamp'] = t_pre
                            f_out.save(deepcopy(current_row))
                            current_row['dvs_frame'][:,:,:] = 0
                            t_pre += args.binsize
                else:
                    # fixed event count mode
                    num_samples = np.ceil(-float(num_evts + ev_count)/args.binsize)
                    for _ in xrange(int(num_samples)):
                        n = min(int(-args.binsize - ev_count), num_evts - offset)
                        sel = slice(offset, offset + n)
                        current_row['dvs_frame'] += raster_evts(d['data'][sel])
                        if sel.stop > sel.start:
                            current_row['timestamp'] = times[sel].mean()
                        offset += n
                        ev_count += n
                        if ev_count == -args.binsize:
                            # SYSTEM Timestamp version:
                            current_row['timestamp'] = (sys_ts - tstart * 1e-6)
                            f_out.save(deepcopy(current_row))
                            current_row['dvs_frame'][:,:,:] = 0
                            ev_count = 0
            pbar_curr = int((sys_ts - tstart * 1e-6) / args.update_prog_every)
            if pbar_curr > pbar_next:
                pbar.update(args.update_prog_every)
                pbar_next = pbar_curr
        pbar.close()
        print('[DEBUG] sys_ts/tstop', sys_ts, tstop*1e-6)
        m.exit.set()
        f_out.exit.set()
        f_out.join()
        while not m.done.is_set():
            print('[DEBUG] waiting for merger')
            time.sleep(1)
        print('[DEBUG] merger done')
        f_in.join()
        print('[DEBUG] stream joined')
        m.join()
        print('[DEBUG] merger joined')
        filesize = os.path.getsize(outfile)
        print('Finished.  Wrote {:.1f}MiB to {}.'.format(filesize/1024**2, outfile))
        
    time.sleep(1)
    os._exit(0)
