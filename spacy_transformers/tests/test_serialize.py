from spacy import Language
from spacy.util import make_tempdir

from spacy_transformers import TransformerData
import srsly


def test_serialize_transformer_data():
    data = {"x": TransformerData.empty()}
    bytes_data = srsly.msgpack_dumps(data)
    new_data = srsly.msgpack_loads(bytes_data)
    assert isinstance(new_data["x"], TransformerData)


def test_transformer_tobytes():
    nlp = Language()
    trf = nlp.add_pipe("transformer")
    trf_bytes = trf.to_bytes()

    nlp2 = Language()
    trf2 = nlp2.add_pipe("transformer")
    trf2.from_bytes(trf_bytes)


def test_transformer_model_tobytes():
    nlp = Language()
    trf = nlp.add_pipe("transformer")
    nlp.initialize()
    trf_bytes = trf.to_bytes()

    nlp2 = Language()
    trf2 = nlp2.add_pipe("transformer")
    trf2.from_bytes(trf_bytes)


def test_transformer_model_todisk():
    nlp = Language()
    trf = nlp.add_pipe("transformer")
    nlp.initialize()
    with make_tempdir() as d:
        trf.to_disk(d)
        nlp2 = Language()
        trf2 = nlp2.add_pipe("transformer")
        trf2.from_disk(d)


def test_transformer_pipeline_tobytes():
    nlp = Language()
    nlp.add_pipe("transformer")
    nlp.initialize()
    assert nlp.pipe_names == ["transformer"]
    nlp_bytes = nlp.to_bytes()

    nlp2 = Language()
    nlp2.add_pipe("transformer")
    nlp2.from_bytes(nlp_bytes)
    assert nlp2.pipe_names == ["transformer"]


def test_transformer_pipeline_todisk():
    nlp = Language()
    nlp.add_pipe("transformer")
    nlp.initialize()
    with make_tempdir() as d:
        nlp.to_disk(d)
        nlp2 = Language()
        nlp2.from_disk(d)
