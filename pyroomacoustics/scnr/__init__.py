"""
Single Channel Noise Reduction
============================

Collection of single channel noise reduction (SCNR) algorithms for speech.
At the moment, only a spectral subtraction method is implemented, similar to [1].

At the following repository, a deep learning approach in Python can be found: https://github.com/santi-pdp/segan

Other methods for speech enhancement / noise reduction employ Wiener filtering [2] and subspace approaches [3].

References
----------

.. [1] M. Berouti, R. Schwartz, and J. Makhoul, *Enhancement of speech corrupted by acoustic noise,*
    ICASSP '79. IEEE International Conference on Acoustics, Speech, and Signal Processing, 1979, pp. 208-211.

.. [2] J. Lim and A. Oppenheim, *All-Pole Modeling of Degraded Speech,*
    IEEE Transactions on Acoustics, Speech, and Signal Processing 26.3 (1978): 197-210.

.. [3] Y. Ephraim and H. L. Van Trees, *A signal subspace approach for speech enhancement,*
    IEEE Transactions on Speech and Audio Processing, vol. 3, no. 4, pp. 251-266, Jul 1995.

"""

from .spectral_subtraction import *
