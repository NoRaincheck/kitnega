from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence, Tuple, Union

_UNK = "<UNK>"


class KneserNeyModel:
    """Interpolated Kneser-Ney language model.

    Matches NLTK's ``KneserNey`` smoothing + ``InterpolatedLanguageModel`` scoring.

    Parameters
    ----------
    order : int
        Maximum n-gram order.
    vocab : list of str, optional
        Known vocabulary. OOV words are mapped to ``<UNK>``.
        Automatically inferred from training data when not provided or empty.
    discount : float, default 0.1
        Absolute discount applied at each order.

    Example
    -------
    ::

        model = KneserNeyModel(order=3)
        model.fit([["the", "cat", "sat"], ["the", "dog", "ran"]])
        model.score("cat", ("the",))   # 0.487...
        model.perplexity("the cat sat")  # 1.8...
    """

    def __init__(self, order: int, vocab: List[str] | None = None, discount: float = 0.1) -> None:
        if not (0 <= discount <= 1):
            raise ValueError("discount must be between 0 and 1")
        if order < 1:
            raise ValueError("order must be >= 1")

        self.order = order
        self.discount = discount
        self._set_vocab(vocab or [])

        self._unigrams: Counter = Counter()
        self._by_order: Dict[int, Dict[Tuple[str, ...], Counter]] = {}

    def _set_vocab(self, words: Sequence[str]) -> None:
        raw = set(words) | {_UNK}
        self.vocab_list = sorted(raw)
        self.vocab_set = raw

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        text_sentences: Union[Sequence[str], Sequence[Sequence[str]]],
        vocab: Sequence[str] | None = None,
    ) -> KneserNeyModel:
        """Fit on raw text sentences.

        Accepts sentences already tokenised (``list[list[str]]``) or
        raw strings that are whitespace-tokenised automatically
        (``list[str]``).

        When no vocabulary has been provided (either at construction or via
        the *vocab* parameter), the vocabulary is inferred from the training
        data.  ``<UNK>`` is always included.

        Parameters
        ----------
        text_sentences : list[list[str]] or list[str]
            Tokenised sentences, or raw strings that are split on whitespace.
        vocab : list of str, optional
            Explicit vocabulary override.

        Returns
        -------
        self

        Example
        -------
        ::

            model = KneserNeyModel(order=2)
            model.fit(["the cat sat", "the dog ran"])     # auto-tokenised
            model.fit([["the", "cat"], ["the", "dog"]])    # already tokenised
        """
        if text_sentences and isinstance(text_sentences[0], str):
            text_sentences = [s.split() for s in text_sentences]  # type: ignore[union-attr]

        if vocab is not None:
            self._set_vocab(vocab)
        elif not self.vocab_list or self.vocab_list == [_UNK]:
            words: set = set()
            for sentence in text_sentences:
                for w in sentence:
                    words.add(w)
            self._set_vocab(words)

        for sentence in text_sentences:
            for n in range(1, self.order + 1):
                for i in range(len(sentence) - n + 1):
                    self._count((sentence[i],) if n == 1 else tuple(sentence[i : i + n]))
        return self

    def _count(self, ngram: Tuple[str, ...]) -> None:
        n = len(ngram)
        if n == 1:
            self._unigrams[ngram[0]] += 1
        else:
            context, word = ngram[:-1], ngram[-1]
            if n not in self._by_order:
                self._by_order[n] = {}
            ctx_dist = self._by_order[n]
            if context not in ctx_dist:
                ctx_dist[context] = Counter()
            ctx_dist[context][word] += 1

    # ------------------------------------------------------------------
    # Public scoring
    # ------------------------------------------------------------------

    def score(self, word: str, context: Tuple[str, ...] | str | None = None) -> float:
        """Smoothed probability P(word | context).

        Out-of-vocabulary words are mapped to ``<UNK>``.

        Parameters
        ----------
        word : str
        context : tuple of str, str, or None, default None
            Preceding n-1 tokens.  ``None`` means the unigram context.

        Returns
        -------
        float

        Example
        -------
        ::

            model.score("cat", ("the",))   # 0.36...
            model.score("cat")              # unigram context
            model.score("unknown_word")     # mapped to <UNK>
        """
        word = word if word in self.vocab_set else _UNK
        return self._score(word, self._normalize_context(context))

    @staticmethod
    def _normalize_context(context):
        if context is None:
            return ()
        if isinstance(context, str):
            return (context,)
        return tuple(context)

    # ------------------------------------------------------------------
    # Internal scoring (no OOV masking)
    # ------------------------------------------------------------------

    def _score(self, word: str, context: Tuple[str, ...]) -> float:
        if not context:
            return self._unigram_score(word)
        if len(context) > self.order - 1:
            context = context[-(self.order - 1) :]
        alpha, gamma = self._alpha_gamma(word, context)
        backoff = self._score(word, context[1:] if len(context) > 1 else ())
        return alpha + gamma * backoff

    # ------------------------------------------------------------------
    # Kneser-Ney smoothing
    # ------------------------------------------------------------------

    def _unigram_score(self, word: str) -> float:
        wc, tc = self._continuation_counts(word, ())
        if tc > 0:
            return wc / tc
        return 1.0 / len(self.vocab_list) if self.vocab_list else 0.0

    def _alpha_gamma(self, word: str, context: Tuple[str, ...]) -> Tuple[float, float]:
        order = len(context) + 1
        prefix = self._prefix_counts(context)

        if order == self.order:
            wc = prefix.get(word, 0)
            tc = sum(prefix.values())
        else:
            wc, tc = self._continuation_counts(word, context)

        alpha = max(wc - self.discount, 0.0) / tc if tc > 0 else 0.0
        n_plus = sum(1 for v in prefix.values() if v > 0)
        gamma = (self.discount * n_plus) / tc if tc > 0 else 1.0
        return alpha, gamma

    def _continuation_counts(self, word: str, context: Tuple[str, ...]) -> Tuple[int, int]:
        higher = len(context) + 2
        if higher > self.order:
            return 0, 0
        ngrams = self._by_order.get(higher, {})
        wc = 0
        total = 0
        for prefix_ngram, counts in ngrams.items():
            if prefix_ngram[1:] == context:
                wc += int(counts.get(word, 0) > 0)
                total += sum(1 for v in counts.values() if v > 0)
        return wc, total

    def _prefix_counts(self, context: Tuple[str, ...]) -> Counter:
        order = len(context) + 1
        if order == 1:
            return self._unigrams
        return self._by_order.get(order, {}).get(context, Counter())

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def perplexity(self, texts: Union[str, Sequence[Union[str, Sequence[str]]]]) -> float:
        """Cross-entropy perplexity on a corpus.

        Accepts:

        - a single string (whitespace-tokenised into one sentence)
        - a list of strings (each whitespace-tokenised)
        - a list of token lists (already tokenised)

        Returns
        -------
        float

        Example
        -------
        ::

            model.perplexity("the cat sat")          # single string
            model.perplexity(["the cat", "a dog"])    # list of strings
            model.perplexity([["the", "cat"]])        # already tokenised
        """
        sentences = self._to_sentences(texts)
        log_probs = []
        for sentence in sentences:
            for i, word in enumerate(sentence):
                ctx = sentence[max(0, i - self.order + 1) : i]
                p = self.score(word, tuple(ctx))
                if p > 0:
                    log_probs.append(math.log2(p))
        if not log_probs:
            return float("inf")
        return math.pow(2.0, -sum(log_probs) / len(log_probs))

    @staticmethod
    def _to_sentences(texts: Union[str, Sequence[Union[str, Sequence[str]]]]) -> List[List[str]]:
        if isinstance(texts, str):
            return [texts.split()]
        if not texts:
            return []
        first = texts[0]
        if isinstance(first, str):
            return [s.split() for s in texts]  # type: ignore[union-attr]
        return texts  # already tokenised

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def get_counts(self) -> dict:
        result: dict = {"unigrams": dict(self._unigrams)}
        for order in sorted(self._by_order):
            result[str(order)] = {}
            for ctx, dist in sorted(self._by_order[order].items(), key=lambda x: x[0]):
                result[str(order)][" ".join(ctx)] = dict(dist)
        return result

    def get_vocab(self) -> List[str]:
        return list(self.vocab_list)


class CharacterKneserNeyModel:
    """Character-level Kneser-Ney language model.

    Tokenises strings into characters and learns the character vocabulary
    from the training data automatically.  Thin wrapper around
    ``KneserNeyModel``.

    Parameters
    ----------
    order : int
        Maximum character n-gram order.  Typical choices: 4-6.
    discount : float, default 0.1
    """

    def __init__(self, order: int = 4, discount: float = 0.1) -> None:
        self._order = order
        self._discount = discount
        self._model: KneserNeyModel | None = None

    @staticmethod
    def _to_chars(text: str) -> List[str]:
        return list(text)

    def fit(self, texts: Sequence[str]) -> CharacterKneserNeyModel:
        """Fit on a corpus of strings (character-level tokenisation)."""
        char_sentences: List[List[str]] = [self._to_chars(t) for t in texts]
        self._model = KneserNeyModel(order=self._order, discount=self._discount)
        self._model.fit(char_sentences)
        return self

    def score(self, char: str, context: str | None = None) -> float:
        """Probability of the next character given a string context.

        Parameters
        ----------
        char : str
            A single character (or string — only the last char is checked
            against the model; other chars provide continuation context).
        context : str, optional
            Preceding characters.  Passed directly as character context.

        Returns
        -------
        float
        """
        ctx_tuple: Tuple[str, ...] = ()
        if context:
            ctx_tuple = tuple(context)
        return self._model.score(char, ctx_tuple)  # type: ignore[union-attr]

    def perplexity(self, texts: Union[str, Sequence[str]]) -> float:
        """Perplexity on a corpus of strings.

        Accepts a single string or a list of strings.
        """
        if isinstance(texts, str):
            texts = [texts]
        char_sentences = [self._to_chars(t) for t in texts]
        return self._model.perplexity(char_sentences)  # type: ignore[union-attr]

    @property
    def vocab_list(self) -> List[str]:
        return self._model.vocab_list  # type: ignore[union-attr]

    @property
    def vocab_set(self) -> set:
        return self._model.vocab_set  # type: ignore[union-attr]

    @property
    def inner(self) -> KneserNeyModel:
        return self._model  # type: ignore[return-value]
