"""MP-30 m19 harness: capture EVERYTHING a piece of code logs or prints and look for prompt-derived words in it.

No comfy import (the unit suite forbids it); the integration test in ``tests/integration/test_privacy_logs_tiny.py``
reuses it. Every record is searched in full: the formatted message, the raw ``msg`` and ``args`` (a ``%s`` argument
is a leak even when a format string hides it), the exception text and the logger name.
"""
import contextlib
import io
import logging
import re

# Prompt-derived words. Each field of the test prompt carries one unique word, so a leak names the field it came from.
SENTINELS = {
    "positive": "zqpositivesentinel",
    "negative": "zqnegativesentinel",
    "identity": "zqidentitysentinel",
    "wardrobe": "zqwardrobesentinel",
    "scene": "zqscenesentinel",
    "pose": "zqposesentinel",
    "camera": "zqcamerasentinel",
    "lighting": "zqlightingsentinel",
    "forbidden_changes": "zqforbiddensentinel",
    "trigger": "zqtriggersentinel",
}
# The conflict colours are prompt words too (m10): the red/blue dress conflict must name fields, not colours.
COLOUR_WORDS = ("red", "blue")


def prompt_fields():
    """Keyword arguments for ``MossPhotonPromptLayout.encode`` (after ``clip``), a red/blue dress conflict inside."""
    s = SENTINELS
    return dict(
        positive="a woman in a red dress, %s" % s["positive"],
        negative="blurry, %s" % s["negative"],
        identity="Anna, %s" % s["identity"],
        wardrobe="a blue dress, %s" % s["wardrobe"],
        scene="a kitchen, %s" % s["scene"],
        pose="standing, %s" % s["pose"],
        camera="35mm, %s" % s["camera"],
        lighting="soft window light, %s" % s["lighting"],
        forbidden_changes="no hats, %s" % s["forbidden_changes"],
    )


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(level=1)
        self.chunks = []

    def emit(self, record):
        try:
            text = record.getMessage()
        except Exception as exc:  # noqa: BLE001 - a broken format string is still evidence of what was passed
            text = "unformattable: %s" % exc
        self.chunks.append("%s|%s|%r|%r|%s" % (record.name, text, record.msg, record.args,
                                                logging.Formatter().formatException(record.exc_info)
                                                if record.exc_info else ""))


class Capture(contextlib.AbstractContextManager):
    """``with Capture() as cap: ...``; ``cap.text`` is everything logged at ANY level plus stdout and stderr."""

    def __init__(self):
        self._handler = _Collect()
        self._out, self._err = io.StringIO(), io.StringIO()
        self._stack = contextlib.ExitStack()

    def __enter__(self):
        root = logging.getLogger()
        old_root, old_disable = root.level, logging.root.manager.disable
        root.setLevel(1)
        logging.disable(logging.NOTSET)
        others = root.handlers[:]  # the run's own console output (and ComfyUI's debug chatter) stays out of the log
        for handler in others:
            root.removeHandler(handler)
        root.addHandler(self._handler)
        # every logger of the pack: a child may sit at WARNING or above
        named = [lg for name, lg in logging.root.manager.loggerDict.items()
                 if isinstance(lg, logging.Logger) and name.split(".")[0] == "MossPhoton"]
        named.append(logging.getLogger("MossPhoton"))
        old_named = [lg.level for lg in named]
        for lg in named:
            lg.setLevel(1)

        def restore():
            root.removeHandler(self._handler)
            for handler in others:
                root.addHandler(handler)
            root.setLevel(old_root)
            logging.disable(old_disable)
            for lg, level in zip(named, old_named):
                lg.setLevel(level)
        self._stack.callback(restore)
        self._stack.enter_context(contextlib.redirect_stdout(self._out))
        self._stack.enter_context(contextlib.redirect_stderr(self._err))
        return self

    def __exit__(self, *exc):
        self._stack.close()
        return False

    @property
    def records(self):
        return list(self._handler.chunks)

    @property
    def text(self):
        return "\n".join(self._handler.chunks + [self._out.getvalue(), self._err.getvalue()])


def leaked_words(text, words):
    """The ``words`` that occur in ``text`` as whole words, case-insensitive."""
    return sorted(w for w in set(words) if re.search(r"(?<![a-z0-9_])%s(?![a-z0-9_])" % re.escape(w), text, re.I))
