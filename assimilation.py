
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator as rgi



def rmse(x,y):
    return np.sqrt(np.nanmean(np.square(x-y)))
def convert_model_time(model_time):
    model_time = int(model_time)
    model_datetime =  datetime(2020, 1, 1) + timedelta(hours=model_time)
    model_time = (np.datetime64(model_datetime) - np.datetime64('1981-01-01T00:00:00')).astype('timedelta64[s]').astype(float) / 3600
    return model_time


def get_distance(time1,lon1, lat1, time2,lon2, lat2):
    s1=50 #km
    t1=0.5 #30min

    x1 = lat1 / 180.0 * np.pi
    x2 = lat2 / 180.0 * np.pi
    y1 = lon1 / 180.0 * np.pi
    y2 = lon2 / 180.0 * np.pi

    cos_distance = np.sin(x1) * np.sin(x2) + np.cos(x1) * np.cos(x2) * np.cos(y1 - y2)
    cos_distance = np.clip(cos_distance, -1, 1)

    distance_km = 6371 * np.arccos(cos_distance)

    distance_t= time1-time2

    distance=np.sqrt(np.square(distance_km/s1)+np.square(distance_t/t1))


    return distance



def process_point(lat_index, lon_index, model_predict, model_times, grid_lon, grid_lat,lon,lat, f3d, cci_swhlist, cci_lonlist, cci_latlist, cci_timelist, window_size):
    if (model_predict[:,lat_index, lon_index] == 0).all():
        return None

    lon_min = max(lon_index - window_size, 0)
    lon_max = min(lon_index + window_size, lon - 1)
    lat_min = max(lat_index - window_size, 0)
    lat_max = min(lat_index + window_size, lat - 1)

    lon_value = grid_lon[lon_index]
    lat_value = grid_lat[lat_index]
    time_value = model_times[-1]

    relid = (cci_lonlist < grid_lon[lon_max]) & (cci_lonlist > grid_lon[lon_min]) & \
            (cci_latlist < grid_lat[lat_max]) & (cci_latlist > grid_lat[lat_min])
    cci_swh_r = cci_swhlist[relid]
    lonlist_r = cci_lonlist[relid]
    latlist_r = cci_latlist[relid]
    timelist_r = cci_timelist[relid]

    if len(cci_swh_r) == 0:
        return None

    swh_fi = f3d(np.array([timelist_r, latlist_r, lonlist_r]).T)
    mask = ~np.isnan(swh_fi) & (swh_fi != 0)

    if (mask==False).all():
        return None

    di_all = get_distance(time_value, lon_value, lat_value, timelist_r[mask], lonlist_r[mask], latlist_r[mask])
    swh_fi = swh_fi[mask]
    swh_oi = cci_swh_r[mask]

    if len(di_all) >= 1:
        min_di_R = min(di_all)
        sum_of_weights = np.sum(np.exp(-di_all ** 2 / (2 * min_di_R ** 2)))
        wi_all = np.exp(-di_all ** 2 / (2 * min_di_R ** 2)) / sum_of_weights
        new_value = model_predict[-1, lat_index, lon_index] + np.sum(
            wi_all * (swh_oi - swh_fi))
        return lat_index, lon_index, new_value
    return None

def oa_assimilation_multi_thread(cci_swhlist, cci_latlist, cci_lonlist, cci_timelist, model_predict, model_time, lat=281, lon=720, window_size=3):
    count = 0

    model_predict = model_predict[:, ::-1, :]

    model_time = np.arange(model_time - model_predict.shape[0] + 1, model_time + 1, 1)
    model_times = np.array([convert_model_time(t) for t in model_time])

    time_min = np.min(model_times)
    time_max = np.max(model_times)
    time_indices = np.where((cci_timelist >= time_min) & (cci_timelist <= time_max))[0]


    cci_swhlist = cci_swhlist[time_indices]
    cci_lonlist = cci_lonlist[time_indices]
    cci_latlist = cci_latlist[time_indices]
    cci_timelist = cci_timelist[time_indices]

    grid_lat = np.linspace(-70, 70, 281)
    grid_lon = np.linspace(-180, 179.5, 720)

    f3d = rgi((model_times, grid_lat, grid_lon), model_predict, method='nearest')

    Fcref = model_predict[-1].copy()

    random_number = np.random.randint(2)

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = []
        for lat_index in range(random_number, lat, 2):
            for lon_index in range( random_number, lon, 2):
                futures.append(executor.submit(process_point, lat_index, lon_index, model_predict, model_times, grid_lon, grid_lat,lon,lat ,f3d, cci_swhlist, cci_lonlist, cci_latlist, cci_timelist, window_size))

        for future in futures:
            result = future.result()
            if result:
                lat_index, lon_index, new_value = result
                Fcref[lat_index, lon_index] = new_value
                count += 1

    end_time = time.time()
    # print(f"Total points processed: {count}")
    # print(f"Total time taken: {end_time - start_time:.2f} seconds")

    return Fcref[::-1, :].copy()


