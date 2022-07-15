"""Response Guided Dimensionality Reduction."""


class RGDR:
    """Response Guided Dimensionality Reduction."""
    def __init__(self, timeseries, lag_shift: int = 0):
        """Instantiate an RGDR operator."""
        self.timeseries = timeseries
        self.lag_shift = lag_shift

    def lag_shifting(self):
        """loop through data with lag shift upto the given `lag_shift` value."""
        # To do: loop through lags
        # To do: call `map_analysis` and `map_regions` and manage
        #  inputs and outputs.
        raise NotImplementedError

    def _map_analysis(self):
        """Perform map analysis.
        
        Use chosen method from `map_analysis` and perform map analysis.
        """
        raise NotImplementedError

    def _clustering_regions(self):
        """Perform regions clustering.

        Use chosen method from `map_regions` and perform clustering of regions
        based on the results from `map_analysis`.
        """
        raise NotImplementedError

    def fit(self, data):
        """Perform RGDR calculations with given data."""
        raise NotImplementedError
