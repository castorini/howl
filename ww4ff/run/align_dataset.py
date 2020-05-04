from multiprocessing import Lock, Process
import logging

from deepspeech_training.util.flags import create_flags, FLAGS
from tqdm import tqdm
import tensorflow as tf

from ww4ff.asr import compute_raw_scores, TranscriptionAligner
from ww4ff.data.dataset import AudioClipDatasetLoader, AudioClipDatasetMetadataWriter, AlignedAudioClipMetadata
from ww4ff.settings import SETTINGS


def work(idx, transcriptions, paths, dataset_path, set_type, lock: Lock):
    with AudioClipDatasetMetadataWriter(dataset_path, set_type, 'aligned-', mode='a') as writer:
        for text, result, path in zip(transcriptions,
                                      tqdm(compute_raw_scores(list(map(str, paths))),total=len(paths), position=idx),
                                      paths):
            try:
                aligned = TranscriptionAligner().align(result, text)
            except:
                logging.warning(f'Transcription skipped {path}')
                continue
            lock.acquire()
            try:
                writer.write(AlignedAudioClipMetadata(path=path, transcription=aligned))
            finally:
                lock.release()


def main(_):
    loader = AudioClipDatasetLoader()
    ds_kwargs = dict(sr=SETTINGS.audio.sample_rate, mono=SETTINGS.audio.use_mono)
    train_ds, dev_ds, test_ds = loader.load_splits(SETTINGS.dataset.dataset_path, **ds_kwargs)
    num_workers = FLAGS.num_workers
    lock = Lock()
    refresh_length = 20  # might need to be smaller for computers with less RAM

    for ds in train_ds, dev_ds, test_ds:
        paths = [m.path for m in ds.metadata_list]
        # TODO: remove dirty fix
        transcriptions = [m.transcription.lower().replace('firefox', 'fire fox') for m in ds.metadata_list]
        transcript_chunks = []
        path_chunks = []
        chunk_len = len(transcriptions) // num_workers
        for idx in range(num_workers):
            last = None if idx == num_workers - 1 else (idx + 1) * chunk_len
            transcript_chunks.append(transcriptions[idx * chunk_len:last])
            path_chunks.append(paths[idx * chunk_len:last])

        while transcript_chunks:
            processes = [Process(target=work, args=(idx,
                                                    t[:refresh_length],
                                                    p[:refresh_length],
                                                    SETTINGS.dataset.dataset_path,
                                                    ds.set_type,
                                                    lock))
                         for idx, (t, p) in enumerate(zip(transcript_chunks, path_chunks))]
            for process in processes:
                process.start()
            for process in processes:
                process.join()
            transcript_chunks = [x[refresh_length:] for x in transcript_chunks]
            transcript_chunks = list(filter(lambda x: len(x) > 0, transcript_chunks))
            path_chunks = [x[refresh_length:] for x in path_chunks]
            path_chunks = list(filter(lambda x: len(x) > 0, path_chunks))


if __name__ == '__main__':
    create_flags()
    tf.app.flags.DEFINE_integer('num_workers', 24, 'Number of workers to use.')
    tf.app.flags.DEFINE_string('src', '', 'Source path to an audio file or directory or catalog file.'
                                          'Catalog files should be formatted from DSAlign. A directory will'
                                          'be recursively searched for audio. If --dst not set, transcription logs (.tlog) will be '
                                          'written in-place using the source filenames with '
                                          'suffix ".tlog" instead of ".wav".')
    tf.app.flags.DEFINE_string('dst', '', 'path for writing the transcription log or logs (.tlog). '
                                          'If --src is a directory, this one also has to be a directory '
                                          'and the required sub-dir tree of --src will get replicated.')
    tf.app.flags.DEFINE_boolean('recursive', False, 'scan dir of audio recursively')
    tf.app.flags.DEFINE_boolean('force', False, 'Forces re-transcribing and overwriting of already existing '
                                                'transcription logs (.tlog)')
    tf.app.flags.DEFINE_integer('vad_aggressiveness', 3, 'How aggressive (0=lowest, 3=highest) the VAD should '
                                                         'split audio')
    tf.app.flags.DEFINE_integer('batch_size', 40, 'Default batch size')
    tf.app.flags.DEFINE_float('outlier_duration_ms', 10000, 'Duration in ms after which samples are considered outliers')
    tf.app.flags.DEFINE_integer('outlier_batch_size', 1, 'Batch size for duration outliers (defaults to 1)')
    tf.app.run(main)
