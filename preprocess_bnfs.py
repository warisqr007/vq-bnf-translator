from utils.argutils import print_args
from pathlib import Path
from itertools import chain
from functools import partial
from multiprocessing.pool import Pool
from tqdm import tqdm
from data_objects.kaldi_interface import KaldiInterface
import librosa
import argparse
import numpy as np
from utils.load_yaml import HpsYaml


def preprocess_bnfs(dataset_root: Path, out_dir: Path, n_processes: int,
                           skip_existing: bool, hparams):
    input_dirs = [dataset_root]
    print("\n    ".join(map(str, ["Using data from:"] + input_dirs)))
    assert all(input_dir.exists() for input_dir in input_dirs)

    # Create the output directories for each output file type
    out_dir.joinpath("bnfs").mkdir(exist_ok=True)


    # # Create a metadata file
    metadata_fpath = out_dir.joinpath("meta_data.txt")
    metadata_file = metadata_fpath.open("a" if skip_existing else "w", encoding="utf-8")

    # Preprocess the dataset
    speaker_dirs = list(chain.from_iterable(input_dir.glob("*") for input_dir in input_dirs))
    func = partial(preprocess_speaker, out_dir=out_dir, skip_existing=skip_existing, hparams=hparams)
    job = Pool(n_processes).imap(func, speaker_dirs)
    for speaker_metadata in tqdm(job, "L2ARCTIC", len(speaker_dirs), unit="speakers"):
        for metadatum in speaker_metadata:
            metadata_file.write("|".join(str(x) for x in metadatum) + "\n")
    metadata_file.close()

    # Verify the contents of the metadata file
    with metadata_fpath.open("r", encoding="utf-8") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
    print("The dataset consists of %d utterances, %d mel frames, %d audio timesteps (%.2f hours)." %
          (len(metadata)))
    # print("Max input length (text chars): %d" % max(len(m[5]) for m in metadata))
    # print("Max mel frames length: %d" % max(int(m[4]) for m in metadata))
    # print("Max audio timesteps length: %d" % max(int(m[3]) for m in metadata))

def preprocess_speaker(speaker_dir, out_dir: Path, skip_existing: bool, hparams):
    metadata = []
    kaldi_dir = speaker_dir.joinpath('kaldi')
    ki = KaldiInterface(wav_scp=str(kaldi_dir.joinpath('wav.scp')), bnf_scp=str(kaldi_dir.joinpath('bnf', 'feats.scp')))
    # print(speaker_dir.name)
    # print(speaker_dir)
    source_speaker = speaker_dir.name

    for wav_fpath in speaker_dir.glob("wav/*"):
        assert wav_fpath.exists()
        wav, _ = librosa.load(wav_fpath, hparams.sample_rate)
        if hparams.rescale:
            wav = wav / np.abs(wav).max() * hparams.rescaling_max
        wav, _ = librosa.effects.trim(wav, top_db=25)
        wav_cat_fname = '{}-{}'.format(speaker_dir.name, wav_fpath.stem)

        metadata.append(process_utterance(wav, ki, out_dir, wav_cat_fname, skip_existing, hparams, source_speaker))
    return [m for m in metadata if m is not None]

def process_utterance(wav: np.ndarray, ref_speaker_feat_interface: KaldiInterface, out_dir: Path, basename: str,
                      skip_existing: bool, hparams, source_speaker: str):
    ## FOR REFERENCE:
    # For you not to lose your head if you ever wish to change things here or implement your own
    # synthesizer.
    # - Both the audios and the mel spectrograms are saved as numpy arrays
    # - There is no processing done to the audios that will be saved to disk beyond volume
    #   normalization (in split_on_silences)
    # - However, pre-emphasis is applied to the audios before computing the mel spectrogram. This
    #   is why we re-apply it on the audio on the side of the vocoder.
    # - Librosa pads the waveform before computing the mel spectrogram. Here, the waveform is saved
    #   without extra padding. This means that you won't have an exact relation between the length
    #   of the wav and of the mel spectrogram. See the vocoder data loader.

    # Skip existing utterances if needed
    ppg_fpath = out_dir.joinpath("ppgs", "ppg-%s.npy" % basename)
    if skip_existing and ppg_fpath.exists():
        return None

    # Skip utterances that are too short
    if len(wav) < hparams.utterance_min_duration * hparams.sample_rate:
        return None

    # Compute ppg
    ppg = ref_speaker_feat_interface.get_feature(source_speaker+'_'+basename.split('-')[-1], 'bnf')
    # ppg_frames = ppg.shape[0]

    # Sometimes ppg can be 1 frame longer than mel
    # min_frames = min(mel_frames, ppg_frames)
    # mel_spectrogram = mel_spectrogram[:, :min_frames]
    # ppg = ppg[:min_frames, :]

    # Write the ppg to disk
    np.save(ppg_fpath, ppg, allow_pickle=False)

    # Return a tuple describing this training example
    return ppg_fpath.name, len(wav)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesses audio files from datasets, encodes them as mel spectrograms "
                    "and writes them to  the disk. Audio files are also saved, to be used by the "
                    "vocoder for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("dataset_root", type=Path, help=\
        "Path to L2-ARCTIC dataset.")
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
        "Path to the output directory that will contain the mel spectrograms, the audios and the "
        "embeds. Defaults to <datasets_root>/SV2TTS/synthesizer/")
    parser.add_argument("-n", "--n_processes", type=int, default=16, help=\
        "Number of processes in parallel.")
    parser.add_argument("-s", "--skip_existing", action="store_true", help=\
        "Whether to overwrite existing files with the same name. Useful if the preprocessing was "
        "interrupted.")
    parser.add_argument("--preprocess_hparams", type=str, default="data_objects/preprocess_hparams.yaml", help=\
        "Path to preprocess config, e.g., data_objects/preprocess_hparams.yaml")
    args = parser.parse_args()
    
    # Process the arguments
    if not hasattr(args, "out_dir"):
        args.out_dir = args.dataset_root.joinpath("SV2TTS", "synthesizer")

    # Create directories
    assert args.dataset_root.exists()
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Preprocess the dataset
    print_args(args, parser)
    hparams = HpsYaml(args.preprocess_hparams)
    args.hparams = hparams
    preprocess_bnfs(**vars(args))
    