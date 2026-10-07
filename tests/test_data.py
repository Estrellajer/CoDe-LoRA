import json

import pytest
from transformers import AutoTokenizer

from codelora.data.collate import CausalLMCollator, Seq2SeqCollator, pretruncate
from codelora.data.tasks import Example, PromptStyle, causal_prompt, load_examples, normalize_answer, score_prediction
from codelora.data.trace import normalize_text, sari, score_trace


def test_official_prompt_uses_family_instruction_and_options(lab):
    (example,) = load_examples("alpha", lab["data"] / "TC/alpha", "train", style=PromptStyle(), limit=1)
    assert example.input_text.startswith("Task:TC\nDataset:alpha\nWhat is the topic of the following paragraph?")
    assert "Option: World, Sports, Business, Science or Technology \n" in example.input_text
    assert example.input_text.endswith("\nAnswer:")
    assert example.source is not None and example.source in example.input_text


def test_prompt_switches_and_copa_has_no_instruction(tmp_path):
    task = tmp_path / "COPA" / "COPA"
    task.mkdir(parents=True)
    (task / "labels.json").write_text('["A", "B"]')
    (task / "train.json").write_text('[{"sentence": "s", "label": "A"}]')
    style = PromptStyle(add_task_name=False, add_dataset_name=False)
    (example,) = load_examples("copa", task, "train", style=style)
    assert example.input_text == "s\nAnswer:"


def test_plain_style_keeps_stripped_source(lab):
    examples = load_examples("alpha", lab["data"] / "TC/alpha", "test", style=None)
    assert all(e.input_text == e.input_text.strip() and e.source.endswith("\n") for e in examples)


def test_limit_is_a_seeded_subset_in_original_order(lab):
    full = load_examples("alpha", lab["data"] / "TC/alpha", "train")
    a = load_examples("alpha", lab["data"] / "TC/alpha", "train", limit=10, seed=3)
    b = load_examples("alpha", lab["data"] / "TC/alpha", "train", limit=10, seed=3)
    assert a == b and len(a) == 10
    assert [full.index(e) for e in a] == sorted(full.index(e) for e in a)
    assert a != load_examples("alpha", lab["data"] / "TC/alpha", "train", limit=10, seed=4)


def test_test_exclusions_are_applied_to_test_split_only(tmp_path):
    task = tmp_path / "TC" / "x"
    task.mkdir(parents=True)
    records = [{"sentence": f"s{i}", "label": "a"} for i in range(5)]
    for split in ("train", "test"):
        (task / f"{split}.json").write_text(json.dumps(records))
    (task / "test_exclusions.json").write_text(json.dumps({"excluded_records": [{"index": 1}, {"index": 3}]}))
    assert len(load_examples("x", task, "train")) == 5
    assert [e.source for e in load_examples("x", task, "test")] == ["s0", "s2", "s4"]


def test_answer_normalisation_and_scoring():
    assert normalize_answer("  Hello,\nWorld! ") == "hello, world!"
    assert normalize_answer("Hello, World!", remove_punctuation=True) == "hello world"
    example = Example("p", "Science or Technology.", "t")
    assert score_prediction(example, "science or technology", official=True) == 1.0
    assert score_prediction(example, "science or technology", official=False) == 0.0
    assert score_prediction(example, "Science  or Technology.", official=False) == 1.0


def test_causal_prompt_adds_one_answer_cue():
    assert causal_prompt("question") == "question\nAnswer: "
    assert causal_prompt("question\nAnswer:") == "question\nAnswer: "


def test_seq2seq_collator_masks_padding(lab):
    tok = AutoTokenizer.from_pretrained(lab["t5"])
    batch = Seq2SeqCollator(tok, 32, 4)([Example("w1 w2 w3", "cat", "t"), Example("w1", "dog or bird or cat", "t")])
    assert batch["labels"].shape[1] <= 4
    assert (batch["labels"] == -100).any() and (batch["labels"] != -100).any()
    assert batch["input_ids"].shape == batch["attention_mask"].shape


def test_causal_collator_masks_prompt_and_appends_eos(lab):
    tok = AutoTokenizer.from_pretrained(lab["qwen"])
    tok.pad_token = tok.eos_token
    collator = CausalLMCollator(tok, 16, 4)
    batch = collator([Example("w1 w2", "cat", "t"), Example("w3 w4 w5 w6 w7", "dog", "t")])
    for ids, labels in zip(batch["input_ids"], batch["labels"], strict=True):
        supervised = ids[labels != -100]
        assert supervised[-1] == tok.eos_token_id  # the answer ends with EOS
        assert (labels != -100).sum() >= 2
    assert (batch["labels"][:, 0] == -100).all()  # the prompt is never supervised


def test_causal_collator_truncates_prompts_on_the_left(lab):
    tok = AutoTokenizer.from_pretrained(lab["qwen"])
    tok.pad_token = tok.eos_token
    tok.truncation_side = "left"
    collator = CausalLMCollator(tok, 6, 3)
    long_prompt = " ".join(f"w{i}" for i in range(30)) + "\nAnswer:"
    batch = collator([Example(long_prompt, "cat", "t")])
    tail = tok(causal_prompt(long_prompt))["input_ids"][-6:]
    assert batch["input_ids"][0][:6].tolist() == tail  # the answer cue survives, the head is dropped


def test_pretruncate_round_trip(lab):
    tok = AutoTokenizer.from_pretrained(lab["t5"])
    text = " ".join(f"w{i}" for i in range(50))
    (cut,) = pretruncate(tok, [text], 10)
    assert len(tok(cut)["input_ids"]) <= 11 and cut != text
    assert pretruncate(tok, ["w1 w2"], 10) == ["w1 w2"]


def test_trace_metrics():
    assert score_trace("a", "a", "accuracy") == 1.0 and score_trace("", "", "accuracy") == 0.0
    assert score_trace("B. no", "B. yes", "scienceqa_accuracy") == 1.0
    assert score_trace(" Hello  World ", "hello world", "exact_match") == 1.0
    assert normalize_text("  A  b ") == "a b"
    assert score_trace("the cat sat", "the cat sat", "rouge_l_f1") == pytest.approx(1.0)
    assert score_trace("x = <NUM_LIT>", "x = 0", "edit_similarity") == 1.0
    assert sari("the cat sat on the mat", "the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0)
    assert 0.0 <= sari("a b c d", "a b", "a b c") <= 1.0
