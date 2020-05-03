from dataclasses import dataclass
from typing import List, Tuple, Iterable
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from deepspeech_training.util.audio import AudioFile
from deepspeech_training.util.config import Config, initialize_globals
from deepspeech_training.util.feeding import split_audio_file
from deepspeech_training.util.flags import create_flags, FLAGS
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1.logging as tflogging
tflogging.set_verbosity(tflogging.ERROR)
import logging
logging.getLogger('sox').setLevel(logging.ERROR)


__all__ = ['AsrOutput', 'compute_raw_scores']


@dataclass
class AsrOutput:
    start_ms: int
    end_ms: int
    probs: np.ndarray


def compute_raw_scores(audio_paths: List[str]) -> Iterable[List[AsrOutput]]:
    from deepspeech_training.train import create_model  # pylint: disable=cyclic-import,import-outside-toplevel
    from deepspeech_training.util.checkpoints import load_graph_for_evaluation
    initialize_globals()
    with tf.Session(config=Config.session_config) as session:
        tf.train.get_or_create_global_step()
        for idx, audio_path in enumerate(audio_paths):
            with AudioFile(audio_path, as_path=True) as wav_path:
                data_set = split_audio_file(wav_path,
                                            batch_size=FLAGS.batch_size,
                                            aggressiveness=1,
                                            outlier_duration_ms=FLAGS.outlier_duration_ms,
                                            outlier_batch_size=FLAGS.outlier_batch_size)
                iterator = tf.data.Iterator.from_structure(data_set.output_types, data_set.output_shapes,
                                                           output_classes=data_set.output_classes)
                batch_time_start, batch_time_end, batch_x, batch_x_len = iterator.get_next()
                no_dropout = [None] * 6
                logits, _ = create_model(batch_x=batch_x, seq_length=batch_x_len, dropout=no_dropout)
                transposed = tf.nn.softmax(tf.transpose(logits, [1, 0, 2]))
                session.run(iterator.make_initializer(data_set))
                if idx == 0:
                    load_graph_for_evaluation(session)
                output_list = []
                while True:
                    try:
                        starts, ends, batch_logits, batch_lengths = session.run([batch_time_start, batch_time_end, transposed, batch_x_len])
                    except tf.errors.OutOfRangeError:
                        break
                    for start, end, logits, length in zip(starts, ends, batch_logits, batch_lengths):
                        output_list.append(AsrOutput(start, end, logits[:length]))
                tf.get_variable_scope().reuse_variables()
            yield output_list
