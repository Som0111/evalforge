from evalforge.scorers import embedding
from evalforge.scorers.embedding import relevance_to_input, semantic_similarity

TEXT = (
    "The city council voted on Tuesday to extend the bus network to the northern suburbs. "
    "The new routes will run every fifteen minutes during the day and connect residents to the "
    "central train station, and the project is funded by a regional transport grant."
)
SUMMARY = "The council approved extending the bus network to the northern suburbs, funded by a transport grant."


def test_identical_strings_are_near_one():
    assert semantic_similarity("the cat sat", "the cat sat") > 0.99


def test_unrelated_strings_are_low():
    assert semantic_similarity("the cat sat", "quarterly revenue declined") < 0.3


def test_summary_is_relevant_to_its_text():
    assert relevance_to_input(SUMMARY, TEXT) > 0.6


def test_bad_input_scores_zero():
    assert semantic_similarity("", "x") == 0.0
    assert semantic_similarity(None, "x") == 0.0


def test_model_loads_once():
    embedding._model.cache_clear()
    semantic_similarity("a b", "c d")
    semantic_similarity("e f", "g h")
    info = embedding._model.cache_info()
    assert (info.misses, info.hits) == (1, 1)
