class BiDAFTokenizer:

    def __init__(self, word_vocab: set, word2idx: dict, idx2word: dict, char_tokenizer: bool = False, max_length: int = 128):
        """
        BiDAF Tokenizer.
        Supports word tokenization and character tokenization.
        Args:
            word_vocab(set): The set of words. For the case of character tokenizer this si the set of the characters.
            word2idx(dict): Word to integer mapping. For the case of char tokenizer this is integer to character.
            idx2word(dict): Integer to word mapping. For the case of char tokenizer this is integer to character.
            char_tokenizer(bool): Flag for char tokenization.
            max_length(int): Maximum word tokens.
        """
        self.word_vocab = word_vocab
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.char_tokenizer = char_tokenizer
        self.max_length = max_length

    def transform(self, words):
        # TODO: Use max length
        tokens = []

        if not self.char_tokenizer:
            for i, word in enumerate(words):
                if word in self.word2idx:
                    tokens.append(self.word2idx[word])
                else:
                    tokens.append(0)
        else:
            for i, word in enumerate(words):
                word_tokens = []
                for char in word:
                    if char in self.word2idx:
                        word_tokens.append(self.word2idx[char])
                    else:
                        word_tokens.append(0)
                tokens.append(word_tokens)

        return tokens