def random_strat(n):
  return None

def timeseries_split(n):
  return None


ALL_METHODS = {
    "leave_n_out": s2s.traintest.leave_n_out,
    "randstrat": s2s.traintest.rand_strat,
    "random": s2s.traintest.random,
    "split": s2s.traintest.split,
    "timeseriessplit": s2s.traintest.timeseries_split,
    "repeated_kfold": s2s.traintest.repeated_kfold,
}
