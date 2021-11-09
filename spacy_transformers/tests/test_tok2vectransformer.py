import pytest
from spacy.training.example import Example
from spacy.util import make_tempdir
from spacy import util
from thinc.api import Model, Config
from .util import _assert_equal_tensors

# fmt: off
TRAIN_DATA = [
    ("I like green eggs", {"tags": ["N", "V", "J", "N"], "cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}}),
    ("Eat blue ham", {"tags": ["V", "J", "N"], "cats": {"POSITIVE": 0.0, "NEGATIVE": 1.0}}),
]


cfg_string_tagger = """
    [nlp]
    lang = "en"
    pipeline = ["tagger"]

    [components]

    [components.tagger]
    factory = "tagger"

    [components.tagger.model]
    @architectures = "spacy.Tagger.v1"
    nO = null

    [components.tagger.model.tok2vec]
    @architectures = "spacy-transformers.Tok2VecTransformer.v1"
    name = "roberta-base"
    tokenizer_config = {"use_fast": false}
    grad_factor = 1.0

    [components.tagger.model.tok2vec.get_spans]
    @span_getters = "spacy-transformers.strided_spans.v1"
    window = 256
    stride = 256

    [components.tagger.model.tok2vec.pooling]
    @layers = "reduce_mean.v1"
    """


cfg_string_textcat = """
    [nlp]
    lang = "en"
    pipeline = ["textcat"]

    [components]

    [components.textcat]
    factory = "textcat"

    [components.textcat.model]
    @architectures = "spacy.TextCatEnsemble.v2"
    nO = null
    
    [components.textcat.model.linear_model]
    @architectures = "spacy.TextCatBOW.v2"

    [components.textcat.model.tok2vec]
    @architectures = "spacy-transformers.Tok2VecTransformer.v1"
    name = "roberta-base"
    tokenizer_config = {"use_fast": false}
    grad_factor = 1.0

    [components.textcat.model.tok2vec.get_spans]
    @span_getters = "spacy-transformers.strided_spans.v1"
    window = 256
    stride = 96

    [components.textcat.model.tok2vec.pooling]
    @layers = "reduce_mean.v1"
    """
# fmt: on


def test_transformer_pipeline_textcat_internal():
    """Test that a textcat with internal transformer runs and trains properly"""
    orig_config = Config().from_str(cfg_string_textcat)
    nlp = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    assert nlp.pipe_names == ["textcat"]
    textcat = nlp.get_pipe("textcat")
    textcat_trf = textcat.model.get_ref("tok2vec").layers[0]
    assert isinstance(textcat_trf, Model)
    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
        for tag in t[1]["cats"]:
            textcat.add_label(tag)

    optimizer = nlp.initialize(lambda: train_examples)
    for i in range(2):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)

    doc = nlp("We're interested at underwater basket weaving.")
    doc_tensor = textcat_trf.predict([doc])

    # ensure IO goes OK
    with make_tempdir() as d:
        file_path = d / "trained_nlp"
        nlp.to_disk(file_path)

        # results are not the same if we don't call from_disk
        nlp2 = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
        nlp2.initialize(lambda: train_examples)
        doc2 = nlp2("We're interested at underwater basket weaving.")
        textcat2 = nlp2.get_pipe("tagger")
        textcat_trf2 = textcat2.model.get_ref("tok2vec").layers[0]
        doc_tensor2 = textcat_trf2.predict([doc2])
        with pytest.raises(AssertionError):
            _assert_equal_tensors(
                doc_tensor2.doc_data[0].tensors, doc_tensor.doc_data[0].tensors
            )

        # results ARE the same if we call from_disk
        nlp3 = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
        nlp3.from_disk(file_path)
        doc3 = nlp3("We're interested at underwater basket weaving.")
        textcat3 = nlp3.get_pipe("tagger")
        textcat_trf3 = textcat3.model.get_ref("tok2vec").layers[0]
        doc_tensor3 = textcat_trf3.predict([doc3])
        _assert_equal_tensors(
            doc_tensor3.doc_data[0].tensors, doc_tensor.doc_data[0].tensors
        )
