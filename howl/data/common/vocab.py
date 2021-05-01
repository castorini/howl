from typing import List, Mapping, Tuple, Union

__all__ = ["Vocab"]


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
        node, word_left = self._nearest_node(word.lower(), node=self.root)
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

    def max_split(self, tokens: str) -> Tuple[str, str]:
        node = self.root
        counter = 0
        for tok in tokens.lower():
            node, word_left = self._nearest_node(tok, node)
            if word_left:
                break
            counter += 1
        if not node.terminal:
            counter = 0
        return tokens[:counter], tokens[counter:]


class Vocab:
    def __init__(
        self, word2idx: Union[Mapping[str, int], List[str]], oov_token_id: int = None, oov_word_repr: str = "[OOV]"
    ):
        if isinstance(word2idx, List):
            word2idx = {word: idx for idx, word in enumerate(word2idx)}
        self.word2idx = {k.lower(): v for k, v in word2idx.items()}
        self.idx2word = {v: k for k, v in word2idx.items()}
        self.oov_token_id = oov_token_id
        self.oov_word_repr = oov_word_repr
        self.trie = VocabTrie()
        for word in self.word2idx:
            self.trie.add_word(word.lower())

    def __len__(self):
        return len(self.word2idx)

    def __getitem__(self, item: Union[str, int]) -> Union[str, int]:
        if isinstance(item, str):
            ret = self.word2idx.get(item.lower(), self.oov_token_id)
        else:
            ret = self.idx2word.get(item, self.oov_word_repr)
        if ret is None:
            raise ValueError(f"couldn't find token for {item}")
        return ret
