from pathlib import Path
import logging
import time

import pyaudio
import numpy as np
import torch

from .args import ArgumentParserBuilder, opt
from howl.data.transform import ZmuvTransform
from howl.settings import SETTINGS
from howl.model import RegisteredModel, Workspace
from howl.model.inference import FrameInferenceEngine


class InferenceClient:
    def __init__(self,
                 engine: FrameInferenceEngine,
                 device: torch.device,
                 words,
                 chunk_size: int = 500):
        self.engine = engine
        self.chunk_size = chunk_size
        self._audio = pyaudio.PyAudio()
        self.words = words
        chosen_idx = 0
        for idx in range(self._audio.get_device_count()):
            info = self._audio.get_device_info_by_index(idx)
            if info['name'] == 'pulse':
                chosen_idx = idx
                break
        stream = self._audio.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=16000,
                                  input=True,
                                  input_device_index=chosen_idx,
                                  frames_per_buffer=self.chunk_size,
                                  stream_callback=self._on_audio)
        self.last_data = np.zeros(self.chunk_size)
        self._audio_buf = []
        self.device = device
        self.stream = stream
        stream.start_stream()

    def join(self):
        while self.stream.is_active():
            time.sleep(0.1)

    def _on_audio(self, in_data, frame_count, time_info, status):
        data_ok = (in_data, pyaudio.paContinue)
        self.last_data = in_data
        self._audio_buf.append(in_data)
        if len(self._audio_buf) != 16:
            return data_ok
        audio_data = b''.join(self._audio_buf)
        self._audio_buf = self._audio_buf[2:]
        arr = np.frombuffer(audio_data, dtype=np.int16).astype(np.float) / 32767
        inp = torch.from_numpy(arr).float().to(self.device)
        self.engine.append_label(self.engine.ingest_frame(inp))
        if self.engine.sequence_present():
            phrase = ' '.join(self.words[x] for x in self.engine.sequence).title()
            print(f'{phrase} detected', end='\r')
        else:
            print('                                ', end='\r')
        return data_ok


def main():
    apb = ArgumentParserBuilder()
    apb.add_options(opt('--model', type=str, choices=RegisteredModel.registered_names(), default='las'),
                    opt('--workspace', type=str, default=str(Path('workspaces') / 'default')),
                    opt('--vocab', type=str, nargs='+', default=['hey', 'firefox']))
    args = apb.parser.parse_args()

    ws = Workspace(Path(args.workspace), delete_existing=False)

    device = torch.device(SETTINGS.training.device)
    zmuv_transform = ZmuvTransform().to(device)
    model = RegisteredModel.find_registered_class(args.model)().to(device).eval()
    zmuv_transform.load_state_dict(torch.load(str(ws.path / 'zmuv.pt.bin')))

    ws.load_model(model, best=False)
    engine = FrameInferenceEngine(model, zmuv_transform, len(args.vocab))
    print(f'Using {engine.settings.make_wakeword(args.vocab)}')

    client = InferenceClient(engine, device, args.vocab)
    client.join()


if __name__ == '__main__':
    main()
