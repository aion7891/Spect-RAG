from spectrafuse.utils.entities import add_entities, clean_phrase, extract_entities


def test_names_products_and_years():
    entities = extract_entities("In 2019 Maya Chen joined Helios Energy and built the Aurora S1.")
    for expected in ["2019", "maya chen", "helios energy", "aurora s1", "helios", "s1"]:
        assert expected in entities


def test_single_word_at_sentence_start_is_skipped():
    entities = extract_entities("Farmers check the readings. Water is scarce.")
    assert entities == []


def test_names_at_sentence_start_with_two_words_are_kept():
    assert "helios energy" in extract_entities("Helios Energy builds solar farms.")


def test_stopwords_are_trimmed():
    assert clean_phrase("The Aurora S1") == "aurora s1"
    assert "section" not in extract_entities("It is described in Section 2.1 of the guide.")


def test_headings_and_lines_stay_separate():
    entities = extract_entities("## 1. History\nAurora Labs was founded in Lisbon.")
    assert "history" not in entities                # heading word at line start
    assert "aurora labs" in entities
    assert not any("history" in e for e in entities)


def test_add_entities_with_custom_function():
    chunks = [{"text": "anything"}]
    add_entities(chunks, entity_fn=lambda text: ["b", "a", "a"])
    assert chunks[0]["entities"] == ["a", "b"]


def test_months_single_letters_and_new_are_not_entities():
    entities = extract_entities("He met them in New Delhi in October 2023. The U.S. team won.")
    assert "october" not in entities and "new" not in entities
    assert "u" not in entities and "s" not in entities
    assert "delhi" in entities and "2023" in entities
