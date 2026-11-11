"""NoLBERT-Calls: NoLBERT adapted to US earnings calls up to a cutoff (hindsight-free reader).

    from nolbert_calls import NoLBERTCalls      # the released model (nolbert_calls/model.py)

Training, evaluation and tables: nolbert_calls.run, nolbert_calls.evaluate, nolbert_calls.report.
Transcript text is licensed (S&P Capital IQ via WRDS): nothing here prints or ships it."""
from .model import NoLBERTCalls  # noqa: F401
