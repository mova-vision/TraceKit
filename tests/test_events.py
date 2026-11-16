from tracekit.core.events import Event, EventEmitter


def test_event_emitter_calls_update_handlers():
    emitter = EventEmitter()
    seen: list[Event] = []

    emitter.on("update", lambda payload: seen.append(payload))
    emitter.emit("update", Event(name="update"))

    assert [e.name for e in seen] == ["update"]


def test_event_emitter_supports_multiple_handlers():
    emitter = EventEmitter()
    count = {"n": 0}

    emitter.on("update", lambda e: count.__setitem__("n", count["n"] + 1))
    emitter.on("update", lambda e: count.__setitem__("n", count["n"] + 1))
    emitter.emit("update", Event(name="update"))

    assert count["n"] == 2
