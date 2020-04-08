import spacy
import os

from spacy_fastlang import LanguageDetector

en_text = "Life is like a box of chocolates. You never know what you're gonna get."


def test_detect_doc_language():
    nlp = spacy.blank("xx")
    nlp.add_pipe(LanguageDetector())
    doc = nlp(en_text)

    assert doc._.language == "en"
    assert doc._.language_score >= 0.8


def test_use_fallback_value_if_under_thresold():
    nlp = spacy.blank("xx")
    nlp.add_pipe(LanguageDetector(threshold=0.99))
    doc = nlp(en_text)

    assert doc._.language == "xx"
    assert doc._.language_score >= 0.8


def test_use_fallback_value_if_language_not_supported():
    nlp = spacy.blank("xx")
    nlp.add_pipe(LanguageDetector(supported_languages=["fr"]))
    doc = nlp(en_text)

    assert doc._.language == "xx"
    assert doc._.language_score >= 0.8


def test_use_custom_fallback():
    nlp = spacy.blank("xx")
    nlp.add_pipe(LanguageDetector(threshold=0.99, default_language="fr"))
    doc = nlp(en_text)

    assert doc._.language == "fr"
    assert doc._.language_score >= 0.8


def test_use_custom_model():
    nlp = spacy.blank("xx")
    nlp.add_pipe(
        LanguageDetector(
            model_path=os.path.realpath(
                os.path.join(__file__, "..", "..", "spacy_fastlang", "lid.176.ftz")
            )
        )
    )
    doc = nlp(en_text)

    assert doc._.language == "en"
    assert doc._.language_score >= 0.8


def test_batch_predictions():
    nlp = spacy.blank("xx")
    nlp.add_pipe(LanguageDetector())
    for doc in nlp.pipe([en_text, en_text]):
        assert doc._.language == "en"
        assert doc._.language_score >= 0.8
