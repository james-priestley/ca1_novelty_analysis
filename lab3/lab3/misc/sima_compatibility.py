import sima
import pickle

from os.path import join

COMPATIBLE_PROTOCOL = 2
NATIVE_SEQUENCES = set(["Sequence",
                    "_AddWrapperSequence",
                    "_DivideWrapperSequence",
                    "_FloorDivideWrapperSequence",
                    "_IndexedSequence",
                    "_Joined_Sequence",
                    "_MaskedSequence",
                    "_MathWrapperSequence",
                    "_MotionCorrectedSequence",
                    "_MultiplyWrapperSequence",
                    "_Sequence_HDF5",
                    "_Sequence_TIFF_Interleaved",
                    "_Sequence_TIFFs",
                    "_Sequence_constant",
                    "_Sequence_memmap",
                    "_Sequence_ndarray",
                    "_SubtractWrapperSequence",
                    "_TrueDivideWrapperSequence",
                    "_WrapperSequence"])

class SIMACompatibilityMixin:
    def save(self, savedir=None):
        """Save the ImagingDataset to a file."""
        print("Saving in SIMA compatible format...")
        if savedir is None:
            savedir = self.savedir
        self.savedir = savedir

        if self._read_only:
            raise Exception('Cannot save read-only dataset.  Change savedir ' +
                            'to a new directory')
        # Keep this out side the with statement
        # If sequences haven't been loaded yet, need to read sequences.pkl
        
        # Saves as the best-match parent class AVAILABLE FROM VANILLA SIMA
        sequences = []
        for seq in self.sequences:
            seq_dict = seq._todict(savedir)
            native_sima_sequence = get_native_sima_sequence(seq)
            seq_dict['__class__'] = native_sima_sequence
            sequences.append(seq_dict)

        with open(join(savedir, 'sequences.pkl'), 'wb') as f:
            pickle.dump(sequences, f, COMPATIBLE_PROTOCOL)

        with open(join(savedir, 'dataset.pkl'), 'wb') as f:
            pickle.dump(self._todict(), f, COMPATIBLE_PROTOCOL)

def get_native_sima_sequence(custom_sequence):
    """Finds the best match parent sequence which is available
    from vanilla SIMA. 
    """
    seq = custom_sequence.__class__
    while not seq.__name__ in NATIVE_SEQUENCES:
        seq = seq.__base__

    if seq is object:
        raise TypeError("No native sequence found! (are you sure this is a SIMA sequence?)")
    else:
        return seq

def sima_compatible(klass):
    if issubclass(klass, sima.ImagingDataset):
        klass.save = SIMACompatibilityMixin.save
    elif issubclass(klass, sima.Sequence):
        def custom_todict(self, savedir=None):
            d = self._old_todict(savedir)
            d['__class__'] = get_native_sima_sequence(self)
            return d

        klass._old_todict = klass._todict
        klass._todict = custom_todict
    else:
        raise TypeError(f"Not sure how to handle {klass.__name__}. " + \
                f"Is this a Sequence or ImagingDataset?")
    
    return klass

