from factcheck.parsing import extract_json, parse_queries, parse_route, parse_verdict


def test_parse_plain_json():
    verdict = parse_verdict('{"verdict": "SUPPORTED", "reasoning": "ok", "citations": ["P1", "p2"]}')
    assert verdict.label == "SUPPORTED"
    assert verdict.citations == ["P1", "P2"]


def test_parse_fenced_json_with_prose():
    text = 'Here you go:\n```json\n{"verdict": "REFUTED", "reasoning": "no", "citations": ["P1"]}\n```'
    assert parse_verdict(text).label == "REFUTED"


def test_parse_json_embedded_in_text():
    text = 'Answer: {"verdict": "NOT ENOUGH INFO", "reasoning": "", "citations": []} thanks'
    assert parse_verdict(text).label == "NOT_ENOUGH_INFO"


def test_parse_aliases():
    assert parse_verdict('{"verdict": "supports", "citations": ["P1"]}').label == "SUPPORTED"
    assert parse_verdict('{"verdict": "contradict", "citations": ["P1"]}').label == "REFUTED"
    assert parse_verdict('{"verdict": "NEI"}').label == "NOT_ENOUGH_INFO"


def test_parse_invalid_returns_none():
    assert parse_verdict("no json here") is None
    assert parse_verdict('{"verdict": "MAYBE"}') is None
    assert extract_json("[1, 2]") is None


def test_string_citation_is_wrapped():
    assert parse_verdict('{"verdict": "SUPPORTED", "citations": "P3"}').citations == ["P3"]


def test_parse_queries_and_fallback():
    assert parse_queries('{"queries": ["a", "b"]}', "claim") == ["a", "b"]
    assert parse_queries("garbage", "claim") == ["claim"]
    assert parse_queries('{"queries": []}', "claim") == ["claim"]


def test_parse_route():
    assert parse_route("FOLLOWUP") == "followup"
    assert parse_route("follow-up question") == "followup"
    assert parse_route("OTHER") == "other"
    assert parse_route("CLAIM") == "claim"
    assert parse_route("something else") == "claim"


def test_think_block_is_ignored_even_with_braces_inside():
    text = '<think>Maybe {"verdict": "REFUTED"} hmm</think>\n{"verdict": "SUPPORTED", "reasoning": "r", "citations": ["P1"]}'
    assert parse_verdict(text).label == "SUPPORTED"


def test_reasoning_without_opening_tag_is_dropped():
    text = 'I should check the passages. </think> {"verdict": "REFUTED", "reasoning": "r", "citations": ["P2"]}'
    assert parse_verdict(text).label == "REFUTED"


def test_unterminated_think_block_gives_no_verdict():
    assert parse_verdict('<think>still thinking {"verdict": "SUPPORTED"}') is None


def test_prose_with_braces_before_json_is_handled():
    text = 'Use the format {like this}. Final: {"verdict": "NOT_ENOUGH_INFO", "citations": []}'
    assert parse_verdict(text).label == "NOT_ENOUGH_INFO"


def test_nested_objects_return_the_outer_object():
    assert extract_json('prefix {"a": {"b": 1}} suffix') == {"a": {"b": 1}}


def test_last_top_level_object_wins():
    assert extract_json('{"a": 1} then {"a": 2}') == {"a": 2}


def test_route_and_queries_ignore_reasoning():
    assert parse_route("<think>the user asks why</think>FOLLOWUP") == "followup"
    assert parse_queries('<think>ideas</think>{"queries": ["x"]}', "claim") == ["x"]
