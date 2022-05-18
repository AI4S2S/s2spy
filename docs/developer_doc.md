# Developer documentation for the AI4S2S project

This documentation is a handbook for us to keep tracking our thoughts and decisions regarding the API design during software development. Most of our choices are motivated in the [NL eScience Center Guide](https://guide.esciencecenter.nl), specifically the [Python chapter](https://guide.esciencecenter.nl/#/best_practices/language_guides/python) and our [software guide checklist](https://guide.esciencecenter.nl/#/best_practices/checklist).

-------------------

## Time utils
[Link to PR](https://github.com/AI4S2S/ai4s2s/pull/7)


Main design code:

* Essentially, the main datastructure resembles the dataframe (`df_splits`) in the proto repo. However, this dataframe is not particularly intuitive to the user, so it will be hidden from the public API.
* It might be nice to use pandas' interval_range or something like that, so that it is unambiguous whether the time stamps mark the beginning or end of the period they denote. 
* We try to keep it simple and make one method do one thing.

Using the timeindex could then look somewhat like this:

```py
import s2s.time

# Ideally, the constructor (`__init__` method) should be simple.
timeindex = s2s.time.TimeIndex(anchor_date="2020-11-30", freq="7d")
print(timeindex)
>> IntervalIndex([(2019-12-02, 2019-12-09], (2019-12-09, 2019-12-16], (2019-12-16, 2019-12-23], (2019-12-23, 2019-12-30], (2019-12-30, 2020-01-06] ... (2020-10-26, 2020-11-02], (2020-11-02, 2020-11-09], (2020-11-09, 2020-11-16], (2020-11-16, 2020-11-23], (2020-11-23, 2020-11-30]], dtype='interval[datetime64[ns], right]', length=52)

# Instead of passing in the target period, max lag, et cetera directly, we can add additional methods:
timeindex.discard(max_lag=5)
print(timeindex)
>> IntervalIndex([(2020-10-26, 2020-11-02], (2020-11-02, 2020-11-09], (2020-11-09, 2020-11-16], (2020-11-16, 2020-11-23], (2020-11-23, 2020-11-30]], dtype='interval[datetime64[ns], right]', length=5)

# Ideally we want an index that's year-agnostic. One option would be to start with a single year and then map to more:
timeindex.map_years(start=1979, end=2018)
print(timeindex)
>> IntervalIndex([
>>     [(2018-10-26, 2020-11-02], ...,
>>     [(2017-10-26, 2020-11-02], ...,
>>     [(2016-10-26, 2020-11-02], ...,
>>     ...,
>>     [(1979-10-26, 2020-11-02], ...
>> ], shape=(5, 39))

# Resampling any pandas/xarray dataset with a time dimension:
df = pd.DataFrame(...)
timeindex.resample(df, method='mean')
print(df)
>> pd.DataFrame(index=...)  # where the index now matches our timeindex

# Obtaining train/test groups
timeindex.get_train_indices(strategy="leave_n_out", params={"n": 1})
>> IntervalIndex([
>>     [(2016-10-26, 2020-11-02], ...,
>>     [(2013-10-26, 2020-11-02], ...,
>>     [(2010-10-26, 2020-11-02], ...,
>>     ...,
>>     [(1980-10-26, 2020-11-02], ...
>> ], shape=(5, 13))

# An alternative could be to implement timeindex.set_traintest_method(...) and have it fixed.
```

## To be continued
-------------------
