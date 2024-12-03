from .perplexity import perplexity, preprocess_batch as perplexity_preprocess_batch
from .accuracy import accuracy, preprocess_batch as accuracy_preprocess_batch

metric_handlers = {
    "perplexity": {
        "handler": perplexity,
        "preprocess_batch": perplexity_preprocess_batch,
    },
    "accuracy": {
        "handler": accuracy,
        "preprocess_batch": accuracy_preprocess_batch,
    },
}
