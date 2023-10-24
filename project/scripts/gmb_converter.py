import argparse
import random
from pathlib import Path
from typing import Iterable, List, Tuple

from spacy import Language
from spacy.lang.en import English
from spacy.tokens import Doc, DocBin, Span
from spacy.training import iob_to_biluo
from spacy.training.iob_utils import tags_to_entities
from tqdm import tqdm

ANNOTATION_FILENAME = "en.tags"


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Convert the Groningen Memory Bank corpus into training data for spaCy",
)
parser.add_argument(
    "input_path",
    metavar="INPUT",
    help="path to the 'data' directory in the Groningen Memory Bank corpus",
)
parser.add_argument(
    "output_path",
    metavar="OUTPUT",
    help="path to the directory in which the converted training data will be stored",
)
parser.add_argument(
    "--n-docs",
    "-n",
    metavar="NUM_DOCS",
    type=int,
    help="number of documents to convert (default: all)",
)
parser.add_argument(
    "--seed",
    "-s",
    metavar="SEED",
    default=0,
    help="seed for random sampling",
)
parser.add_argument(
    "--splits",
    metavar="SPLITS",
    default="80|10|10",
    help="train, dev and test split percentages (integers), separated by the '|' character (default: 80|10|10)",
)
parser.add_argument(
    "--coalesce-entities",
    action="store_true",
    default=True,
    help="coalesce consecutive named entities that are the same into multi-token spans",
)


def convert_sent_entities_to_iob(
    entities: Iterable[str], *, coalesce_entities: bool, upper_case: bool
) -> List[str]:
    """
    Convert Groningen Meaning Bank entity annotations to IOB
    format.

    :param entities:
        Iterable of entities for a single sentence.
    :param coalesce_entities:
        If ``True``, consecutive named entities that are the same are converted into
        multi-token spans.
    :param upper_case:
        Convert tags to uppercase.
    :returns:
        The converted entities.
    """

    if not coalesce_entities:
        return [
            f"B-{ent.upper() if upper_case else ent}" if ent != "O" else ent
            for ent in entities
        ]

    converted = []
    last_entity = "O"
    for entity in entities:
        if entity == "O":
            converted.append(entity)
        elif last_entity != entity:
            # New entity begins.
            converted.append(f"B-{entity.upper() if upper_case else entity}")
        else:
            # Old entity continues.
            converted.append(f"I-{entity.upper() if upper_case else entity}")
        last_entity = entity

    return converted


def convert_gmb_tags_to_doc(
    nlp: Language, input_data: str, *, coalesce_entities: bool
) -> Doc:
    """
    Convert a Groningen Meaning Bank annotations file to a spaCy
    Doc instance. The following annotations are converted:
      - Token
      - POS
      - Lemma
      - Named Entity

    Annotation file format: http://svn.ask.it.usyd.edu.au/trac/candc/wiki/IOFormats

    :param nlp:
        The Language instance.
    :input_data:
        String data representing the tag file of a single GMB document.
    :param coalesce_entities:
        If ``True``, consecutive named entities that are the same are converted into
        multi-token spans.
    :returns:
        The ``Doc`` instance.
    """
    words: List[str] = []
    sent_starts: List[bool] = []
    pos_tags: List[str] = []
    lemmas: List[str] = []
    biluo_tags: List[str] = []

    for gmb_sent in input_data.split("\n\n"):
        gmb_sent = gmb_sent.strip()
        if not gmb_sent:
            continue
        lines = [line.strip() for line in gmb_sent.split("\n") if line.strip()]
        cols: List[Tuple[str, ...]] = list(zip(*[line.split("\t") for line in lines]))
        if len(cols) < 4:
            raise ValueError(
                "Annotation data must contain at least the first four columns"
            )
        length = len(cols[0])
        words.extend(cols[0])
        sent_starts.extend([True] + [False] * (length - 1))
        entity_tags = convert_sent_entities_to_iob(
            cols[3],
            coalesce_entities=coalesce_entities,
            upper_case=True,
        )
        biluo_tags.extend(iob_to_biluo(entity_tags))
        pos_tags.extend(cols[1])
        lemmas.extend(cols[2])

    doc = Doc(nlp.vocab, words=words)
    for i, token in enumerate(doc):
        token.tag_ = pos_tags[i]
        token.is_sent_start = sent_starts[i]
        token.lemma_ = lemmas[i]
    entities = tags_to_entities(biluo_tags)
    doc.ents = [
        Span(doc, start=s, end=e + 1, label=L) for L, s, e in entities
    ]  # type: ignore
    return doc


def split_training_sets(
    docs: List[Doc], splits: Tuple[int, int, int]
) -> Tuple[List[Doc], List[Doc], List[Doc]]:
    """
    Split the dataset into train, dev and test sets.

    :param docs:
        Dataset to split.
    :param splits:
        Percentages of the train, dev and test sets respectively.
    :returns:
        The train, dev and test sets respectively.
    """
    n_docs = len(docs)
    assert sum(splits) == 100

    n_train = splits[0] * n_docs // 100
    n_dev = splits[1] * n_docs // 100
    n_test = n_docs - n_train - n_dev

    train = docs[0:n_train]
    dev = docs[n_train : n_train + n_dev]
    test = docs[n_train + n_dev :]

    assert len(train) == n_train and len(dev) == n_dev and len(test) == n_test
    assert not set.intersection(set(train), set(dev), set(test))

    return train, dev, test


def convert_to_docbin(docs: List[Doc], output_filepath: Path):
    """
    Convert the docs to a spaCy `DocBin` file.

    :param docs:
        Docs to convert.
    :param output_filepath:
        Path in which to create the `DocBin` file.
    """
    if not docs:
        return

    db = DocBin(docs=docs, store_user_data=False)
    data = db.to_bytes()

    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    with output_filepath.open("wb") as f:
        f.write(data)


if __name__ == "__main__":
    args = parser.parse_args()

    corpus_path = args.input_path
    output_path = args.output_path
    rng = random.Random(args.seed)
    coalesce_entities = args.coalesce_entities
    n_docs_to_convert = args.n_docs

    if n_docs_to_convert is not None and n_docs_to_convert <= 0:
        raise ValueError(
            f"Number of documents to convert must be a positive, non-zero integer"
        )

    splits = args.splits.split("|")
    if len(splits) != 3:
        raise ValueError(f"Splits '{splits}' is not valid")
    splits = tuple(int(x) for x in splits)
    if sum(splits) != 100:
        raise ValueError(f"Splits '{splits}' do not add up to 100")

    print(f"Input directory: {corpus_path}")
    print(f"Output directory: {output_path}")
    print(f"Splits: {splits}")
    print(f"Number of documents to convert: {n_docs_to_convert}")
    print(f"Coalesce consecutive named entities: {coalesce_entities}")
    print("Processing documents...")

    nlp = English()
    docs = []
    for filepath in tqdm(sorted(Path(corpus_path).rglob(ANNOTATION_FILENAME))):
        with open(str(filepath), encoding="utf-8") as f:
            new_doc = convert_gmb_tags_to_doc(
                nlp, f.read(), coalesce_entities=coalesce_entities
            )
            docs.append(new_doc)

    rng.shuffle(docs)

    if n_docs_to_convert is not None and len(docs) > n_docs_to_convert:
        docs = docs[:n_docs_to_convert]

    print(f"Converting {len(docs)} documents...")

    train_data, dev_data, test_data = split_training_sets(docs, splits)
    convert_to_docbin(train_data, Path(output_path) / "train.spacy")
    convert_to_docbin(dev_data, Path(output_path) / "dev.spacy")
    convert_to_docbin(test_data, Path(output_path) / "test.spacy")

    print(f"Wrote {len(train_data)} documents to the training set")
    print(f"Wrote {len(dev_data)} documents to the development set")
    print(f"Wrote {len(test_data)} documents to the test set")
