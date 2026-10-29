"""The ear's features: MFCC from conf/mfcc.conf and online i-vectors from ivector/.

    from christman_sound.ear.features import Mfcc, Ivectors, SilenceWeighting

    mfcc = Mfcc(model / "conf" / "mfcc.conf").compute(wave)   # (T, 40) float32
    iv = Ivectors(model / "ivector")
    rows = iv.online(mfcc)                                     # (T, 100) float32, period 10

    live = iv.incremental(use_most_recent=True)                # the recognizer's path
    live.update_frame_weights(weights.delta_weights(...))      # silence weighting from the decoder
    live.accept(mfcc_rows); live.get(t); live.finalize()
"""

from .ivectors import Incremental, Ivectors, SilenceWeighting, linear_cg
from .mfcc import Mfcc, MfccStream

__all__ = ["Mfcc", "MfccStream", "Ivectors", "Incremental", "SilenceWeighting", "linear_cg"]
