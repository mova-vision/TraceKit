from tracekit.core.events import EventEmitter


def test_event_emitter_calls_update_handlers():
    emitter = EventEmitter()
    seen: list[str] = []

    emitter.on("update", lambda payload: seen.append(payload))
    emitter.emit("update", "hello")

    assert seen == ["hello"]


def test_event_emitter_supports_multiple_handlers():
    emitter = EventEmitter()
    count = {"n": 0}

    emitter.on("update", lambda: count.__setitem__("n", count["n"] + 1))
    emitter.on("update", lambda: count.__setitem__("n", count["n"] + 1))
    emitter.emit("update")

    assert count["n"] == 2
