from typing import List, Mapping, Union, Tuple


__all__ = ['WakeWordTokenizer', 'TranscriptTokenizer', 'VocabTrie']


class TranscriptTokenizer:
    def encode(self, transcript: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError


class VocabTrie:
    class Node:
        def __init__(self, terminal: bool = True):
            self.terminal = terminal
            self.children = {}

        def __repr__(self):
            return repr(self.children)

    def __init__(self):
        self.root = VocabTrie.Node(terminal=False)

    def __repr__(self):
        return repr(self.root)

    def _nearest_node(self, word: str, node: Node):
        try:
            return self._nearest_node(word[1:], node.children[word[0]])
        except (KeyError, IndexError):
            return node, word

    def add_word(self, word: str):
        node, word_left = self._nearest_node(word, node=self.root)
        if not word_left:
            node.terminal = True
            return
        new_node = VocabTrie.Node(terminal=False)
        node.children[word_left[0]] = new_node
        for character in word_left[1:]:
            node = VocabTrie.Node(terminal=False)
            new_node.children[character] = node
            new_node = node
        new_node.terminal = True

    def max_split(self, tokens: Union[List[str], str]) -> Tuple[Union[List[str], str], Union[List[str], str]]:
        node = self.root
        counter = 0
        for tok in tokens:
            node, word_left = self._nearest_node(tok, node)
            if word_left:
                break
            counter += 1
        if not node.terminal:
            counter = 0
        return tokens[:counter], tokens[counter:]


class Vocab:
    def __init__(self,
                 word2idx: Mapping[str, int],
                 oov_token_id: int = None,
                 oov_word_repr: str = '[OOV]'):
        self.word2idx = word2idx
        self.idx2word = {v: k for k, v in word2idx.items()}
        self.oov_token_id = oov_token_id
        self.oov_word_repr = oov_word_repr

    def __len__(self):
        return len(self.word2idx)

    def __getitem__(self, item: Union[str, int]) -> Union[str, int]:
        if isinstance(item, str):
            ret = self.word2idx.get(item, self.oov_token_id)
        else:
            ret = self.idx2word.get(item, self.oov_word_repr)
        if ret is None:
            raise ValueError('couldn\'t find token')
        return ret

    def to_trie(self) -> VocabTrie:
        trie = VocabTrie()
        for word in self.word2idx:
            trie.add_word(word)
        return trie


class WakeWordTokenizer(TranscriptTokenizer):
    def __init__(self,
                 vocab: Vocab,
                 ignore_oov: bool = True):
        self.vocab = vocab
        self.trie = vocab.to_trie()
        self.ignore_oov = ignore_oov

    def decode(self, ids: List[int]) -> str:
        return ' '.join(self.vocab[id] for id in ids)

    def encode(self, transcript: str) -> List[int]:
        encoded_output = []
        while transcript:
            word, transcript = self.trie.max_split(transcript)
            if word:
                encoded_output.append(self.vocab[word])
            else:
                if not self.ignore_oov:
                    if self.vocab.oov_token_id is None:
                        raise ValueError
                    encoded_output.append(self.vocab.oov_token_id)
                transcript = transcript[1:]
        return encoded_output
