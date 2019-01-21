import numpy as np


class Wiener(object):
    """

    Parameters
    ----------
    frame_len : int
        Frame length in samples.
    win : numpy array
        Window applied in analysis or synthesis.
    alpha : float
        Smoothing factor within [0,1] for updating noise level. Closer to `1`
        gives more weight to the previous noise level, while closer to `0`
        gives more weight to the current frame's level. Closer to `0` can track
        more rapid changes in the noise level. However, if a speech frame is
        incorrectly identified as noise, you can end up removing desired
        speech.
    a_dd : float
        Smoothing factor within [0,1] for updating priori update.
    thresh : float
        Threshold to distinguish between (signal+noise) and (noise) frames. A
        high value will classify more frames as noise but might remove desired
        signal!
    """

    def __init__(self, frame_len, win=None, alpha=0.9, a_dd=0.98, thresh=0.15):

        if frame_len % 2:
            raise ValueError("Frame length should be even as this method "
                             "relies on 50% overlap.")

        self.frame_len = frame_len
        self.alpha = alpha
        self.a_dd = a_dd
        self.thresh = thresh

        # derived parameters and state variables
        self.hop = frame_len // 2
        self.noisy_psd = np.zeros(self.hop + 1, dtype=np.float32)
        self.noise_psd = np.ones(self.hop + 1) * self.thresh / (self.hop + 1)
        self.G = np.ones(self.hop + 1, dtype=np.complex64)
        self.posteri_prev = np.ones(self.hop + 1)

        # normalization factor
        if win is None:
            self.norm_fact = 1
        else:
            self.norm_fact = np.dot(win, win) / frame_len

    def apply(self, frame_dft):
        """
        Compute Wiener filter output in the frequency domain.

        Parameters
        ----------
        frame_dft : numpy array
            DFT of input samples.

        Returns
        -------
        numpy array
            Output of denoising in the frequency domain.
        """

        self.noisy_psd[:] = np.abs(frame_dft) ** 2 / \
                            (self.frame_len * self.norm_fact)

        """ VAD to update noise PSD estimate """
        posteri = self.noisy_psd / self.noise_psd
        posteri_prime = posteri - 1
        posteri_prime[posteri_prime < 0] = 0
        priori = self.a_dd * (self.G ** 2) * self.posteri_prev \
                 + (1 - self.a_dd) * posteri_prime
        log_sigma_k = posteri * priori / (1 + priori) - np.log(1 + priori)
        if sum(log_sigma_k) / self.frame_len < self.thresh:
            self.noise_psd[:] = self.alpha * self.noise_psd + (1 - self.alpha) * self.noisy_psd

        """ compute and apply Wiener filter """
        self.G[:] = np.sqrt(priori / (1 + priori))

        self.posteri_prev[:] = posteri

        return frame_dft * self.G


def apply_wiener(noisy_signal, frame_len=512, win=None, alpha=0.9, a_dd=0.98,
                 thresh=0.15):
    """
    One-shot function to apply iterative Wiener filtering for denoising.

    Parameters
    ----------
    noisy_signal : numpy array
        Real signal in time domain.
    frame_len : int
        Frame length in samples.
    win : numpy array
        Window applied in analysis or synthesis. Default is hanning analysis
        window.
    alpha : float
        Smoothing factor within [0,1] for updating noise level. Closer to `1`
        gives more weight to the previous noise level, while closer to `0`
        gives more weight to the current frame's level. Closer to `0` can track
        more rapid changes in the noise level. However, if a speech frame is
        incorrectly identified as noise, you can end up removing desired
        speech.
    a_dd : float
        Smoothing factor within [0,1] for updating priori update.
    thresh : float
        Threshold to distinguish between (signal+noise) and (noise) frames. A
        high value will classify more frames as noise but might remove desired
        signal!

    Returns
    -------
    numpy array
        Enhanced/denoised signal.
    """

    from pyroomacoustics import hann
    from pyroomacoustics.transform import STFT

    hop = frame_len // 2
    if win is None:
        win = hann(frame_len, flag='asymmetric', length='full')
    stft = STFT(frame_len, hop=hop, analysis_window=win, streaming=True)
    scnr = Wiener(frame_len=frame_len, win=win, alpha=alpha, a_dd=a_dd,
                  thresh=thresh)

    processed_audio = np.zeros_like(noisy_signal)
    n = 0
    while noisy_signal.shape[0] - n >= hop:
        # SCNR in frequency domain
        stft.analysis(noisy_signal[n:(n + hop), ])
        S = scnr.apply(stft.X)

        # back to time domain
        processed_audio[n:n + hop, ] = stft.synthesis(S)

        # update step
        n += hop

    return processed_audio


