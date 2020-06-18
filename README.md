# spacy_fastlang

## Install

Assuming you have a working python environment, you can simply install it using

```
pip install spacy_fastlang
```

## Usage

The library exports a pipeline component called `LanguageDetector` that will set two spacy extensions

- doc.\_.language = ISO code of the detected language or `xx` as a fallback
- doc.\_.language_score = confidence

```
from spacy_fastlang import LanguageDetector
nlp = spacy.load("...")
nlp.add_pipe(LanguageDetector())
doc = nlp(en_text)

doc._.language == "..."
doc._.language_score >= ...
```

## Options

[Check the tests](./tests/test_spacy_fastlang.py) to see more examples and available options

## License

Everythin is under `MIT` except the default model which is distributed under [Creative Commons Attribution-Share-Alike License 3.0](https://creativecommons.org/licenses/by-sa/3.0/) by facebook [here](https://fasttext.cc/docs/en/language-identification.html)
