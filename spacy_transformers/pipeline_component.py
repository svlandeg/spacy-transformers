from typing import List, Callable, Iterable, Iterator, Optional, Dict, Tuple, Union
from spacy.pipeline import Pipe
from spacy.language import Language
from spacy.pipeline.pipe import deserialize_config
from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy.training import Example
from spacy import util
from spacy.util import minibatch
from thinc.api import Model, Config, set_dropout_rate, Optimizer
import srsly
import torch
from transformers import WEIGHTS_NAME, CONFIG_NAME
from pathlib import Path

from .util import huggingface_from_pretrained, batch_by_length
from .annotation_setters import null_annotation_setter
from .data_classes import FullTransformerBatch, TransformerData
from .layers import TransformerListener


DEFAULT_CONFIG_STR = """
[transformer]
max_batch_items = 4096

[transformer.set_extra_annotations]
@annotation_setters = "spacy-transformers.null_annotation_setter.v1"

[transformer.model]
@architectures = "spacy-transformers.TransformerModel.v1"
name = "roberta-base"
tokenizer_config = {"use_fast": true}

[transformer.model.get_spans]
@span_getters = "spacy-transformers.strided_spans.v1"
window = 128
stride = 96
"""

DEFAULT_CONFIG = Config().from_str(DEFAULT_CONFIG_STR)
DOC_EXT_ATTR = "trf_data"


@Language.factory(
    "transformer",
    assigns=[f"doc._.{DOC_EXT_ATTR}"],
    default_config=DEFAULT_CONFIG["transformer"],
)
def make_transformer(
    nlp: Language,
    name: str,
    model: Model[List[Doc], FullTransformerBatch],
    set_extra_annotations: Callable[[List[Doc], FullTransformerBatch], None],
    max_batch_items: int,
):
    """Construct a Transformer component, which lets you plug a model from the
    Huggingface transformers library into spaCy so you can use it in your
    pipeline. One or more subsequent spaCy components can use the transformer
    outputs as features in its model, with gradients backpropagated to the single
    shared weights.

    model (Model[List[Doc], FullTransformerBatch]): A thinc Model object wrapping
        the transformer. Usually you will want to use the TransformerModel
        layer for this.
    set_extra_annotations (Callable[[List[Doc], FullTransformerBatch], None]): A
        callback to set additional information onto the batch of `Doc` objects.
        The doc._.trf_data attribute is set prior to calling the callback.
        By default, no additional annotations are set.
    """
    return Transformer(
        nlp.vocab, model, set_extra_annotations, max_batch_items=max_batch_items, name=name
    )


def install_extensions() -> None:
    if not Doc.has_extension(DOC_EXT_ATTR):
        Doc.set_extension(DOC_EXT_ATTR, default=None)


class Transformer(Pipe):
    """spaCy pipeline component that provides access to a transformer model from
    the Huggingface transformers library. Usually you will connect subsequent
    components to the shared transformer using the TransformerListener layer.
    This works similarly to spaCy's Tok2Vec component and Tok2VecListener
    sublayer.

    The activations from the transformer are saved in the doc._.trf_data extension
    attribute. You can also provide a callback to set additional annotations.

    vocab (Vocab): The Vocab object for the pipeline.
    model (Model[List[Doc], FullTransformerBatch]): A thinc Model object wrapping
        the transformer. Usually you will want to use the TransformerModel
        layer for this.
    set_extra_annotations (Callable[[List[Doc], FullTransformerBatch], None]): A
        callback to set additional information onto the batch of `Doc` objects.
        The doc._.trf_data attribute is set prior to calling the callback.
        By default, no additional annotations are set.
    """

    def __init__(
        self,
        vocab: Vocab,
        model: Model[List[Doc], FullTransformerBatch],
        set_extra_annotations: Callable = null_annotation_setter,
        *,
        name: str = "transformer",
        max_batch_items: int = 128 * 32,  # Max size of padded batch
    ):
        """Initialize the transformer component."""
        self.name = name
        self.vocab = vocab
        self.model = model
        if not isinstance(self.model, Model):
            raise ValueError(f"Expected Thinc Model, got: {type(self.model)}")
        self.set_extra_annotations = set_extra_annotations
        self.cfg = {"max_batch_items": max_batch_items}
        self.listeners: List[TransformerListener] = []
        install_extensions()

    def add_listener(self, listener: TransformerListener) -> None:
        """Add a listener for a downstream component. Usually internals."""
        listener.set_dim("nO", self.model.get_dim("nO"))
        self.listeners.append(listener)

    def find_listeners(self, model: Model) -> None:
        """Walk over a model, looking for layers that are TransformerListener
        subclasses that have an upstream_name that matches this component.
        Listeners can also set their upstream_name attribute to the wildcard
        string '*' to match any `Transformer`.

        You're unlikely to ever need multiple `Transformer` components, so it's
        fine to leave your listeners upstream_name on '*'.
        """
        names = ("*", self.name)
        for node in model.walk():
            if isinstance(node, TransformerListener) and node.upstream_name in names:
                self.add_listener(node)

    def __call__(self, doc: Doc) -> Doc:
        """Apply the pipe to one document. The document is modified in place,
        and returned. This usually happens under the hood when the nlp object
        is called on a text and all components are applied to the Doc.

        docs (Doc): The Doc to preocess.
        RETURNS (Doc): The processed Doc.

        DOCS: https://spacy.io/api/transformer#call
        """
        outputs = self.predict([doc])
        self.set_annotations([doc], outputs)
        return doc

    def pipe(self, stream: Iterable[Doc], *, batch_size: int = 128) -> Iterator[Doc]:
        """Apply the pipe to a stream of documents. This usually happens under
        the hood when the nlp object is called on a text and all components are
        applied to the Doc.

        stream (Iterable[Doc]): A stream of documents.
        batch_size (int): The number of documents to buffer.
        YIELDS (Doc): Processed documents in order.

        DOCS: https://spacy.io/api/transformer#pipe
        """
        for outer_batch in minibatch(stream, batch_size):
            outer_batch = list(outer_batch)
            for indices in batch_by_length(outer_batch, self.cfg["max_batch_items"]):
                subbatch = [outer_batch[i] for i in indices]
                self.set_annotations(subbatch, self.predict(subbatch))
            yield from outer_batch

    def predict(self, docs: Iterable[Doc]) -> FullTransformerBatch:
        """Apply the pipeline's model to a batch of docs, without modifying them.
        Returns the extracted features as the FullTransformerBatch dataclass.

        docs (Iterable[Doc]): The documents to predict.
        RETURNS (FullTransformerBatch): The extracted features.

        DOCS: https://spacy.io/api/transformer#predict
        """
        activations = self.model.predict(docs)
        batch_id = TransformerListener.get_batch_id(docs)
        for listener in self.listeners:
            listener.receive(batch_id, activations.doc_data, None)
        return activations

    def set_annotations(
        self, docs: Iterable[Doc], predictions: FullTransformerBatch
    ) -> None:
        """Assign the extracted features to the Doc objects. By default, the
        TransformerData object is written to the doc._.trf_data attribute. Your
        set_extra_annotations callback is then called, if provided.

        docs (Iterable[Doc]): The documents to modify.
        predictions: (FullTransformerBatch): A batch of activations.

        DOCS: https://spacy.io/api/pipe#set_annotations
        """
        doc_data = list(predictions.doc_data)
        for doc, data in zip(docs, doc_data):
            doc._.trf_data = data
        self.set_extra_annotations(docs, predictions)

    def update(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.0,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
        set_annotations: bool = False,
    ) -> Dict[str, float]:
        """Prepare for an update to the transformer.

        Like the `Tok2Vec` component, the `Transformer` component is unusual
        in that it does not receive "gold standard" annotations to calculate
        a weight update. The optimal output of the transformer data is unknown;
        it's a hidden layer inside the network that is updated by backpropagating
        from output layers.

        The `Transformer` component therefore does not perform a weight update
        during its own `update` method. Instead, it runs its transformer model
        and communicates the output and the backpropagation callback to any
        downstream components that have been connected to it via the
        TransformerListener sublayer. If there are multiple listeners, the last
        layer will actually backprop to the transformer and call the optimizer,
        while the others simply increment the gradients.

        examples (Iterable[Example]):
            A batch of Example objects. Only the `predicted` doc object is used,
            the reference doc is ignored.
        drop (float): The dropout rate.
        set_annotations (bool): Whether or not to update the Example objects
            with the predictions.
        sgd (thinc.api.Optimizer): The optimizer.
        losses (Dict[str, float]): Optional record of the loss during training.
            Updated using the component name as the key.
        RETURNS (Dict[str, float]): The updated losses dictionary.

        DOCS: https://spacy.io/api/transformer#update
        """
        if losses is None:
            losses = {}
        docs = [eg.predicted for eg in examples]
        if isinstance(docs, Doc):
            docs = [docs]
        set_dropout_rate(self.model, drop)
        trf_full, bp_trf_full = self.model.begin_update(docs)
        d_tensors = []
        losses.setdefault(self.name, 0.0)

        def accumulate_gradient(d_trf_datas: List[TransformerData]):
            """Accumulate tok2vec loss and gradient. This is passed as a callback
            to all but the last listener. Only the last one does the backprop.
            """
            nonlocal d_tensors
            for i, d_trf_data in enumerate(d_trf_datas):
                for d_tensor in d_trf_data.tensors:
                    losses[self.name] += float((d_tensor ** 2).sum())  # type: ignore
                if i >= len(d_tensors):
                    d_tensors.append(d_trf_data.tensors)
                else:
                    for j, d_tensor in enumerate(d_trf_data.tensors):
                        d_tensors[i][j] += d_tensor

        def backprop(d_trf_datas: List[TransformerData]):
            """Callback to actually do the backprop. Passed to last listener."""
            nonlocal d_tensors
            accumulate_gradient(d_trf_datas)
            d_trf_full = trf_full.unsplit_by_doc(d_tensors)
            d_docs = bp_trf_full(d_trf_full)
            if sgd is not None:
                self.model.finish_update(sgd)
            d_tensors = []
            return d_docs

        batch_id = TransformerListener.get_batch_id(docs)
        for listener in self.listeners[:-1]:
            listener.receive(batch_id, trf_full.doc_data, accumulate_gradient)
        if self.listeners[:-1]:
            self.listeners[-1].receive(batch_id, trf_full.doc_data, backprop)
        if set_annotations:
            self.set_annotations(docs, trf_full)
        return losses

    def get_loss(self, docs, golds, scores):
        """A noop function, for compatibility with the Pipe API. See the `update`
        method for an explanation of the loss mechanics of the component.
        """
        pass

    def begin_training(
        self,
        get_examples: Callable[[], Iterable[Example]],
        *,
        pipeline: Optional[List[Tuple[str, Callable[[Doc], Doc]]]] = None,
        sgd: Optional[Optimizer] = None,
    ):
        """Initialize the pipe for training, using data examples if available.

        get_examples (Callable[[], Iterable[Example]]): Optional function that
            returns gold-standard Example objects.
        pipeline (List[Tuple[str, Callable]]): Optional list of pipeline
            components that this component is part of. Corresponds to
            nlp.pipeline.
        sgd (thinc.api.Optimizer): Optional optimizer. Will be created with
            create_optimizer if it doesn't exist.
        RETURNS (thinc.api.Optimizer): The optimizer.

        DOCS: https://spacy.io/api/transformer#begin_training
        """
        docs = [Doc(Vocab(), words=["hello"])]
        self.model.initialize(X=docs)
        if pipeline is not None:
            for i, (name1, proc1) in enumerate(pipeline):
                if proc1 is self:
                    for name2, proc2 in pipeline[i:]:
                        if hasattr(proc2, "model"):
                            self.find_listeners(proc2.model)
                    break


    def to_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = tuple()
    ) -> None:
        """Serialize the pipe to disk.

        path (str / Path): Path to a directory.
        exclude (Iterable[str]): String names of serialization fields to exclude.

        DOCS: https://spacy.io/api/transformer#to_disk
        """

        def save_model(p):
            trf_dir = Path(p).absolute()
            if not trf_dir.exists():
                trf_dir.mkdir()
            self.model.attrs["tokenizer"].save_pretrained(str(trf_dir))
            transformer = self.model.layers[0].shims[0]._model
            torch.save(transformer.state_dict(), trf_dir / WEIGHTS_NAME)
            transformer.config.to_json_file(trf_dir / CONFIG_NAME)

        serialize = {}
        serialize["cfg"] = lambda p: srsly.write_json(p, self.cfg)
        serialize["vocab"] = lambda p: self.vocab.to_disk(p)
        serialize["model"] = lambda p: save_model(p)
        util.to_disk(path, serialize, exclude)

    def from_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = tuple()
    ) -> "Transformer":
        """Load the pipe from disk.

        path (str / Path): Path to a directory.
        exclude (Iterable[str]): String names of serialization fields to exclude.
        RETURNS (Transformer): The loaded object.

        DOCS: https://spacy.io/api/transformer#from_disk
        """

        def load_model(p):
            p = Path(p).absolute()
            tokenizer, transformer = huggingface_from_pretrained(
                p, self.model.attrs["tokenizer_config"]
            )
            self.model.attrs["tokenizer"] = tokenizer
            self.model.attrs["set_transformer"](self.model, transformer)

        deserialize = {
            "vocab": self.vocab.from_disk,
            "cfg": lambda p: self.cfg.update(deserialize_config(p)),
            "model": load_model,
        }
        util.from_disk(path, deserialize, exclude)
        return self
