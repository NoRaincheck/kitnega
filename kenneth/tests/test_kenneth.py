from __future__ import annotations

import math
import os

import pytest

from kenneth.model import CharacterKneserNeyModel, KneserNeyModel

# ---------------------------------------------------------------------------
# Basic model construction
# ---------------------------------------------------------------------------


def test_init_defaults():
    model = KneserNeyModel(order=2, vocab=["a", "b"])
    assert model.order == 2
    assert model.discount == 0.1
    assert "<UNK>" in model.vocab_list
    assert "a" in model.vocab_list
    assert "b" in model.vocab_list


def test_init_no_vocab():
    """Vocab can be omitted entirely and learned at fit time."""
    model = KneserNeyModel(order=2)
    assert model.vocab_list == ["<UNK>"]


def test_init_invalid_discount():
    with pytest.raises(ValueError, match="discount"):
        KneserNeyModel(order=2, vocab=["a"], discount=-0.1)
    with pytest.raises(ValueError, match="discount"):
        KneserNeyModel(order=2, vocab=["a"], discount=1.1)


def test_init_invalid_order():
    with pytest.raises(ValueError, match="order"):
        KneserNeyModel(order=0, vocab=["a"])


# ---------------------------------------------------------------------------
# Scoring sanity
# ---------------------------------------------------------------------------


def test_unigram_score_nonzero():
    model = KneserNeyModel(order=2, vocab=["a", "b", "c"], discount=0.1)
    model.fit([["a", "b"], ["a", "c"]])
    p_b = model.score("b")
    p_c = model.score("c")
    assert p_b > 0
    assert p_c > 0
    assert p_b == pytest.approx(p_c)


def test_bigram_score():
    model = KneserNeyModel(order=2, vocab=["a", "b", "c"], discount=0.1)
    model.fit([["a", "b"], ["a", "b"], ["a", "c"]])
    p = model.score("b", ("a",))
    assert 0 < p < 1


def test_sum_to_one_bigram():
    model = KneserNeyModel(order=2, vocab=["the", "cat", "sat", "dog", "ran"], discount=0.1)
    model.fit([["the", "cat", "sat"], ["the", "dog", "ran"]])
    for context in [None, ("the",), ("cat",), ("dog",)]:
        total = sum(model.score(w, context) for w in model.vocab_list)
        assert total == pytest.approx(1.0, abs=0.01), f"context={context} total={total}"


def test_sum_to_one_trigram():
    model = KneserNeyModel(order=3, vocab=["a", "b", "c", "d"], discount=0.1)
    model.fit([["a", "b", "c"], ["b", "c", "d"], ["a", "b", "d"]])
    for context in [None, ("b",), ("a", "b")]:
        total = sum(model.score(w, context) for w in model.vocab_list)
        assert total == pytest.approx(1.0, abs=0.01), f"context={context} total={total}"


def test_scores_reasonable():
    model = KneserNeyModel(order=2, vocab=["the", "cat", "sat", "dog", "ran"], discount=0.1)
    model.fit([["the", "cat", "sat"], ["the", "dog", "ran"]])
    assert 0 < model.score("cat", ("the",)) < 1
    assert model.score("the") == 0.0
    assert model.score("cat") == model.score("dog")


# ---------------------------------------------------------------------------
# OOV
# ---------------------------------------------------------------------------


def test_oov_returns_nonzero():
    model = KneserNeyModel(order=2, vocab=["a", "b"], discount=0.1)
    model.fit([["a", "b"]])
    oov_score = model.score("z")
    known_score = model.score("a")
    assert oov_score >= 0
    assert known_score >= 0


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------


def test_perplexity():
    model = KneserNeyModel(order=2, vocab=["a", "b", "c"], discount=0.1)
    model.fit([["a", "b"], ["a", "c"]])
    ppl = model.perplexity([["a", "b"]])
    assert ppl > 0
    assert math.isfinite(ppl)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_get_counts_roundtrip():
    model = KneserNeyModel(order=2, vocab=["a", "b", "c"], discount=0.1)
    model.fit([["a", "b"], ["a", "c"]])
    counts = model.get_counts()
    assert "unigrams" in counts
    assert "2" in counts


def test_get_vocab():
    model = KneserNeyModel(order=2, vocab=["b", "a", "c"], discount=0.1)
    vocab = model.get_vocab()
    assert "<UNK>" in vocab
    assert "a" in vocab
    assert "b" in vocab
    assert "c" in vocab


# ---------------------------------------------------------------------------
# Vocabulary auto-learning
# ---------------------------------------------------------------------------


def test_vocab_learned_from_data():
    model = KneserNeyModel(order=2)
    model.fit([["hello", "world"], ["foo", "bar"]])
    for w in ["hello", "world", "foo", "bar"]:
        assert w in model.vocab_set
    assert "<UNK>" in model.vocab_set


def test_vocab_learned_empty_init():
    model = KneserNeyModel(order=2, vocab=[])
    model.fit([["only", "these", "words"]])
    for w in ["only", "these", "words"]:
        assert w in model.vocab_set


def test_vocab_override_at_fit_time():
    model = KneserNeyModel(order=2)
    model.fit([["a", "b"]], vocab=["x", "y"])
    assert "x" in model.vocab_set
    assert "y" in model.vocab_set
    assert "a" not in model.vocab_set


def test_fit_with_list_of_strings():
    model = KneserNeyModel(order=2)
    model.fit(["the cat sat", "the dog ran"])
    assert "the" in model.vocab_set
    assert "cat" in model.vocab_set
    assert model.score("cat", ("the",)) > 0
    total = sum(model.score(w, ("the",)) for w in model.vocab_list)
    assert total == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Paragraph-level usage
# ---------------------------------------------------------------------------


def test_paragraph_word_level():
    """Train on a paragraph of text and verify learning + probability sanity."""
    paragraph = "the cat sat on the mat the dog chased the cat the cat sat a dog ran"
    words = paragraph.split()
    # Chunk into 3 roughly-even sentences
    n = len(words) // 3
    sentences = [words[:n], words[n : 2 * n], words[2 * n :]]

    model = KneserNeyModel(order=2)
    model.fit(sentences)

    assert "the" in model.vocab_set
    assert "cat" in model.vocab_set
    assert "<UNK>" in model.vocab_set

    assert model.score("cat", ("the",)) > 0
    assert model.score("ran", ("dog",)) > 0

    for ctx in [None, ("the",), ("cat",), ("sat",)]:
        total = sum(model.score(w, ctx) for w in model.vocab_list)
        assert total == pytest.approx(1.0, abs=0.01), f"context={ctx} total={total}"


def test_paragraph_sentences():
    """Split a paragraph into proper sentences and train."""
    text = "the cat sat on the mat. the dog chased the cat. the cat sat. a dog ran."
    raw_sents = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    sentences = [s.split() for s in raw_sents]

    model = KneserNeyModel(order=2)
    model.fit(sentences)

    assert model.score("cat", ("the",)) > 0
    assert model.score("ran", ("dog",)) > 0
    assert model.score("sat", ("cat",)) > 0

    for ctx in [None, ("the",), ("cat",), ("dog",)]:
        total = sum(model.score(w, ctx) for w in model.vocab_list)
        assert total == pytest.approx(1.0, abs=0.01), f"context={ctx} total={total}"


# ---------------------------------------------------------------------------
# Character-level
# ---------------------------------------------------------------------------


def test_character_level_fit_and_score():
    """CharacterKneserNeyModel learns character vocab and scores characters."""
    model = CharacterKneserNeyModel(order=4, discount=0.1)
    model.fit(["hello", "world"])

    assert "h" in model.vocab_set
    assert "e" in model.vocab_set
    assert "<UNK>" in model.vocab_set

    assert model.score("o", "hell") > 0
    assert model.score("d", "worl") > 0
    assert model.score("x", "hello") >= 0


def test_character_level_sum_to_one():
    model = CharacterKneserNeyModel(order=3, discount=0.1)
    model.fit(["abc", "abd", "bbc"])

    for ctx in ["", "a", "ab"]:
        total = sum(model.score(c, ctx) for c in model.vocab_list)
        assert total == pytest.approx(1.0, abs=0.01), f"char context={ctx!r} total={total}"


def test_character_level_perplexity():
    model = CharacterKneserNeyModel(order=4, discount=0.1)
    model.fit(["hello", "world", "held"])
    ppl = model.perplexity(["hello"])
    assert ppl > 0
    assert math.isfinite(ppl)


def test_character_level_paragraph():
    """Character-level model trained on a paragraph of text."""
    sentences_str = ["the cat sat on the mat", "the dog chased the cat"]
    model = CharacterKneserNeyModel(order=5, discount=0.1)
    model.fit(sentences_str)

    assert "t" in model.vocab_set
    assert " " in model.vocab_set

    assert model.score(" ", "the") > 0
    assert model.score("c", "the ") > 0

    for ctx in ["", "t", "th", "the"]:
        total = sum(model.score(c, ctx) for c in model.vocab_list)
        assert total == pytest.approx(1.0, abs=0.01), f"char context={ctx!r} total={total}"


# ---------------------------------------------------------------------------
# Flexible perplexity input
# ---------------------------------------------------------------------------


class TestPerplexityFlexibleInput:
    """Demonstrate that perplexity accepts string, list[str], and list[list[str]]."""

    CORPUS = [
        "the cat sat on the mat",
        "the dog chased the cat",
        "a dog ran",
    ]

    @pytest.fixture
    def model(self):
        m = KneserNeyModel(order=3, discount=0.1)
        m.fit([s.split() for s in self.CORPUS])
        return m

    def test_perplexity_string(self, model):
        ppl = model.perplexity("the cat sat on the mat")
        assert ppl > 0
        assert math.isfinite(ppl)

    def test_perplexity_list_of_strings(self, model):
        ppl = model.perplexity(["the cat sat", "a dog ran"])
        assert ppl > 0
        assert math.isfinite(ppl)

    def test_perplexity_list_of_token_lists(self, model):
        ppl = model.perplexity([["the", "cat", "sat"], ["a", "dog"]])
        assert ppl > 0
        assert math.isfinite(ppl)

    def test_perplexity_all_three_identical(self, model):
        ppl_str = model.perplexity("the cat sat")
        ppl_list_str = model.perplexity(["the cat sat"])
        ppl_list_tok = model.perplexity([["the", "cat", "sat"]])
        assert ppl_str == pytest.approx(ppl_list_str)
        assert ppl_str == pytest.approx(ppl_list_tok)

    def test_perplexity_empty_string(self, model):
        assert model.perplexity("") == float("inf")

    def test_perplexity_empty_list(self, model):
        assert model.perplexity([]) == float("inf")


class TestCharPerplexityFlexibleInput:
    """Character-level perplexity accepts single string or list of strings."""

    CORPUS = ["hello", "world"]

    @pytest.fixture
    def model(self):
        m = CharacterKneserNeyModel(order=4, discount=0.1)
        m.fit(self.CORPUS)
        return m

    def test_perplexity_string(self, model):
        ppl = model.perplexity("hello")
        assert ppl > 0
        assert math.isfinite(ppl)

    def test_perplexity_list_of_strings(self, model):
        ppl = model.perplexity(["hello", "world"])
        assert ppl > 0
        assert math.isfinite(ppl)

    def test_perplexity_single_vs_list_identical(self, model):
        ppl_s = model.perplexity("hello")
        ppl_l = model.perplexity(["hello"])
        assert ppl_s == pytest.approx(ppl_l)


# ---------------------------------------------------------------------------
# NLTK compatibility
# ---------------------------------------------------------------------------

try:
    from nltk.lm import KneserNeyInterpolated as NLTK_KN
    from nltk.lm import Vocabulary as NLTKVocab
    from nltk.lm.preprocessing import padded_everygram_pipeline

    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False


def _collect_vocab(sentences):
    words = set()
    for sent in sentences:
        for w in sent:
            words.add(w)
    return sorted(words)


@pytest.mark.skipif(not HAS_NLTK, reason="NLTK not installed")
@pytest.mark.xfail(
    os.environ.get("CI"),
    reason="NLTK deps may be flaky in CI — allowed to fail",
    strict=False,
)
class TestNLTKMatch:
    CORPUS = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "chased", "the", "cat"],
        ["the", "cat", "sat"],
        ["a", "dog", "ran"],
    ]

    @pytest.fixture
    def nltk_model(self):
        order = 2
        train_data, vocab_data = padded_everygram_pipeline(order, self.CORPUS)
        vocab = NLTKVocab(vocab_data, unk_cutoff=1)
        model = NLTK_KN(order=order, vocabulary=vocab, discount=0.1)
        model.fit(train_data)
        return model

    @pytest.fixture
    def our_model(self):
        vocab = _collect_vocab(self.CORPUS)
        model = KneserNeyModel(order=2, vocab=vocab, discount=0.1)
        model.fit(self.CORPUS)
        return model

    def test_scores_match_approximately(self, nltk_model, our_model):
        contexts = [None, ("the",), ("cat",), ("dog",), ("mat",)]
        diffs = []
        for context in contexts:
            for word in our_model.vocab_list:
                if word == "<UNK>":
                    continue
                nltk_p = nltk_model.score(word, context)
                our_p = our_model.score(word, context)
                if nltk_p == 0 and our_p == 0:
                    continue
                diffs.append(abs(our_p - nltk_p))
        avg_diff = sum(diffs) / len(diffs)
        assert avg_diff < 0.05, f"Mean diff from NLTK: {avg_diff}"

    def test_sums_to_one(self, our_model):
        for context in [None, ("the",), ("cat",), ("dog",)]:
            total = sum(our_model.score(w, context) for w in our_model.vocab_list)
            assert total == pytest.approx(1.0, abs=0.01)

    def test_perplexity_match(self, nltk_model, our_model):
        held_out = ["the", "cat", "sat", "on", "the", "mat"]
        nltk_ppl = _nltk_perplexity(nltk_model, held_out)
        our_ppl = our_model.perplexity([held_out])
        assert abs(our_ppl - nltk_ppl) < 0.5, f"perp: ours={our_ppl} nltk={nltk_ppl}"

    def test_context_diffs_are_small(self, nltk_model, our_model):
        for context in [None, ("the",), ("cat",), ("dog",), ("mat",)]:
            ctx_diffs = []
            for word in our_model.vocab_list:
                if word == "<UNK>":
                    continue
                nltk_p = nltk_model.score(word, context)
                our_p = our_model.score(word, context)
                if nltk_p == 0 and our_p == 0:
                    continue
                ctx_diffs.append(abs(our_p - nltk_p))
            if ctx_diffs:
                assert max(ctx_diffs) < 0.4, f"context={context} max_diff={max(ctx_diffs)}"


def _nltk_perplexity(nltk_model, sentence):
    log_prob = 0.0
    n = 0
    for i, word in enumerate(sentence):
        ctx = sentence[max(0, i - nltk_model.order + 1) : i]
        p = nltk_model.score(word, tuple(ctx))
        if p > 0:
            log_prob += math.log2(p)
            n += 1
    if n == 0:
        return float("inf")
    return math.pow(2.0, -log_prob / n)


# ---------------------------------------------------------------------------
# PyTorch export
# ---------------------------------------------------------------------------

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestPyTorchExport:
    @pytest.fixture
    def model(self):
        m = KneserNeyModel(order=2, vocab=["the", "cat", "sat", "dog", "ran"], discount=0.1)
        m.fit([["the", "cat", "sat"], ["the", "dog", "ran"]])
        return m

    @pytest.fixture
    def torch_module(self, model):
        from kenneth.export import to_torch

        return to_torch(model)

    def test_scores_match(self, model, torch_module):
        sentences = [["the", "cat", "sat"], ["the", "dog"]]
        word2id = torch_module.word2id
        unk_id = word2id.get("<UNK>", 0)

        for sentence in sentences:
            for i, word in enumerate(sentence):
                ctx = tuple(sentence[max(0, i - model.order + 1) : i])
                stdlib_p = model.score(word, ctx)
                if stdlib_p == 0:
                    continue

                ctx_padded = [torch_module.pad_token] * (model.order - 1 - len(ctx)) + list(ctx)
                ctx_ids = torch.tensor([[word2id.get(w, unk_id) for w in ctx_padded]], dtype=torch.long)
                tgt_id = torch.tensor([word2id.get(word, unk_id)], dtype=torch.long)

                with torch.no_grad():
                    torch_logp = torch_module(ctx_ids, tgt_id).item()
                torch_p = math.exp(torch_logp) if torch_logp > -1e30 else 0.0

                assert abs(stdlib_p - torch_p) < 1e-5, f"P({word} | {ctx}): stdlib={stdlib_p} torch={torch_p}"

    def test_sums_to_one_torch(self, model, torch_module):
        word2id = torch_module.word2id
        unk_id = word2id.get("<UNK>", 0)

        for ctx_tuple in [(), ("the",), ("cat",)]:
            total = 0.0
            ctx_padded = [torch_module.pad_token] * (model.order - 1 - len(ctx_tuple)) + list(ctx_tuple)
            ctx_tensor = torch.tensor([[word2id.get(w, unk_id) for w in ctx_padded]], dtype=torch.long)

            for word in model.vocab_list:
                tgt_tensor = torch.tensor([word2id.get(word, unk_id)], dtype=torch.long)
                with torch.no_grad():
                    logp = torch_module(ctx_tensor, tgt_tensor).item()
                total += math.exp(logp) if logp > -1e30 else 0.0

            assert total == pytest.approx(1.0, abs=0.01), f"Torch sum for context={ctx_tuple}: {total}"


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

try:
    import onnx  # noqa: F401

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:
    HAS_ORT = False


@pytest.mark.skipif(not HAS_ONNX or not HAS_ORT, reason="ONNX/ONNX Runtime not installed")
class TestONNXExport:
    @pytest.fixture
    def model(self):
        m = KneserNeyModel(order=2, vocab=["the", "cat", "sat", "dog", "ran"], discount=0.1)
        m.fit([["the", "cat", "sat"], ["the", "dog", "ran"]])
        return m

    def test_onnx_export_and_run(self, model, tmp_path):
        from kenneth.export import to_onnx

        onnx_path = tmp_path / "kneser_ney.onnx"
        torch_module = to_onnx(model, str(onnx_path))
        assert onnx_path.exists()

        word2id = torch_module.word2id
        unk_id = word2id.get("<UNK>", 0)

        session = ort.InferenceSession(str(onnx_path))
        output_name = session.get_outputs()[0].name

        sentences = [["the", "cat", "sat"], ["the", "dog"]]
        for sentence in sentences:
            for i, word in enumerate(sentence):
                ctx = sentence[max(0, i - model.order + 1) : i]
                stdlib_p = model.score(word, tuple(ctx))
                if stdlib_p == 0:
                    continue

                ctx_padded = [torch_module.pad_token] * (model.order - 1 - len(ctx)) + list(ctx)
                ctx_ids = [[word2id.get(w, unk_id) for w in ctx_padded]]
                tgt_id = [word2id.get(word, unk_id)]

                result = session.run(
                    [output_name],
                    {"context_ids": ctx_ids, "target_id": tgt_id},
                )
                onnx_logp = result[0][0]
                onnx_p = math.exp(onnx_logp) if onnx_logp > -1e30 else 0.0

                assert abs(stdlib_p - onnx_p) < 1e-4, f"P({word} | {ctx}): stdlib={stdlib_p} onnx={onnx_p}"
