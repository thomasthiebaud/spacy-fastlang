import os
import fasttext
from spacy.tokens import Doc


class LanguageDetector(object):
    def __init__(
        self,
        default_language="xx",
        supported_languages=None,
        threshold=0,
        model_path=os.path.realpath(os.path.join(__file__, "..", "lid.176.ftz")),
    ):
        Doc.set_extension("language", default=default_language, force=True)
        Doc.set_extension("language_score", default=0, force=True)

        self.model = fasttext.load_model(model_path)
        self.default_language = default_language
        self.supported_languages = supported_languages
        self.threshold = threshold

    def __call__(self, doc: Doc):
        labels, confidences = self.model.predict(doc.text)
        label = labels[0]
        confidence = confidences[0]

        language = self.default_language
        if confidence > self.threshold:
            language = label[9:]  # label looks like __label__<ISO code of language>

        if (
            self.supported_languages is not None
            and language not in self.supported_languages
        ):
            language = self.default_language

        doc._.language = language
        doc._.language_score = confidence

        return doc
