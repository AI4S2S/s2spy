import s2spy
import xarray as xr
import pandas as pd

# Load data  # TODO: replace with realistic datasets
file_path = '../tests/test_rgdr/test_data'
field = xr.open_dataset(f'{file_path}/sst_daily_1979-2018_5deg_Pacific_175_240E_25_50N.nc')
target = xr.open_dataset(f'{file_path}/tf5_nc5_dendo_80d77.nc')

# Configure time/calendars
cal = s2spy.time.AdventCalendar((8, 31), freq = "30d")
cal = cal.map_to_data(field)

# Resample data
field_resampled = s2spy.time.resample(cal, field)
target_resampled = s2spy.time.resample(cal, target)

# TODO: Resample should return data-array so we don't have to extract it here
target_timeseries = target_resampled.sel(cluster=3).ts.isel(i_interval=0)
precursor_fields = field_resampled.sst

# TODO: move to s2spy
def train_test_split(a, b, test_size, dim="anchor_year"):
    """Train test splitter for n-dimensional (xarray) data."""
    return (a.isel({dim: slice(None, -test_size)}), a.isel({dim: slice(-test_size, None)}),
            b.isel({dim: slice(None, -test_size)}), b.isel({dim: slice(-test_size, None)}))

# hold out some data for final testing
# TODO: later on we want to use an outer cross-validation scheme here as well
precursor_train, precursor_test, target_train, target_test = train_test_split(
    precursor_fields, target_timeseries, test_size=2, dim="anchor_year")

lag_clusters = []
for lag in range(1, 4):
    print(lag)
    rgdr = s2spy.RGDR(target_train)  # TODO: RGDR.fit(precuror_train, target_train)
    clustered_data = rgdr.fit(precursor_train.isel(i_interval=lag))
    clustered_df = clustered_data.to_pandas().T.drop(columns=[0.0])  # TODO: no float as column name
    # TODO: catch "empty" cluster results (those with only a 0.0 column)
    clustered_df.columns = [f"lag_{lag}_cluster_{i}" for i in range(len(clustered_df.columns))]
    lag_clusters.append((lag, rgdr, clustered_df))

# Concat and convert to dataframe
df = pd.concat([x for (a, b, x) in lag_clusters], axis=1)

X = df
y = target_train.to_pandas()

# TODO: add realistic model and parameters
model = ...
parameters = ...
gs = GridSearchCV(model, parameters)
gs.fit(X, y)
