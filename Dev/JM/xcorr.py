import numpy as np

def xcorr(x, y = None, scale='none', maxlag = None):
    """compute correlation of x and y.
    If y not given compute autocorrelation.
    
    Arguments:
        x {np.ndarray} -- signal
        y {np.ndarray} -- signal
    
    Keyword Arguments:
        scale {str} -- can be either "none", "biased" or "unbiased" (default: {'none'})
        maxlag {str} -- maximum lag to be returned. This should be <= round(y.size+x.size-1)/2 (default: {'none'})
    Returns:
        [np.ndarry] -- corresponding lags
        [np.ndarray] -- resulting correlation signal
    """
    # If y is None ccmpute autocorrelation
    if y is None:
        y = x
    # Pad shorter array if signals are different lengths
    else:
        if x.size > y.size:
            pad_amount = x.size - y.size
            y = np.append(y, np.repeat(0, pad_amount))
        elif y.size > x.size:
            pad_amount = y.size - x.size
            x = np.append(x, np.repeat(0, pad_amount))
    if maxlag is None:
        maxlag = (x.size+y.size-1)/2
    if maxlag>round((y.size+x.size-1)/2):
        raise ValueError("maxlag should be <= round(y.size+x.size-1)/2")
    corr = np.correlate(x, y, mode='full')  # scale = 'none'
    lags = np.arange(-maxlag, maxlag + 1)
    # lags = np.arange(-x.size, x.size-1)
    corr = corr[(x.size-1-maxlag):(x.size + maxlag)]
    if scale == 'biased':
        corr = corr / x.size
    elif scale == 'unbiased':
        corr /= (x.size - abs(lags))
    elif scale == 'coeff':
        corr /= np.sqrt(np.dot(x, x) * np.dot(y, y))
    # lags = lags[int(round(len(lags)/2)-maxlag+1) : int(round(len(lags)/2)+maxlag-1)]
    # corr = corr[int(round(len(corr)/2)-maxlag+1) : int(round(len(corr)/2)+maxlag-1)]
    return lags, corr