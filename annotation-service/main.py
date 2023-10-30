from dataclasses import dataclass
from typing import List, Tuple

from fastapi import FastAPI
import spacy
from spacy.tokens import Doc

en_core_web_trf = spacy.load("en_core_web_trf")

app = FastAPI()


@dataclass
class Token:
    text: str
    lemma: str


@dataclass
class Span:
    text: str
    label: str
    span: Tuple[int, int]


@dataclass
class Annotations:
    tokens: List[Token]
    entities: List[Span]


def convert_spacy_doc(doc: Doc):
    tokens = [Token(text=token.text, lemma=token.lemma_) for token in doc]
    entities = [
        Span(text=ent.text, label=ent.label_, span=(ent.start, ent.end))
        for ent in doc.ents
    ]
    return Annotations(tokens=tokens, entities=entities)


@app.post("/annotate")
def annotate(docs: List[str]):
    annotated_docs = en_core_web_trf.pipe(docs)
    return [convert_spacy_doc(doc) for doc in annotated_docs]
