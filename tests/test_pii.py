import pytest

from app.generate.pii import KINDS, Redactor, iban_valid, luhn_valid


def redact(text, kinds=KINDS):
    r = Redactor(kinds)
    return r.redact(text), r.found


def test_emails_are_replaced_and_repeats_share_a_placeholder():
    text, found = redact(
        "Write to Ada.Lovelace@Example.org or bob@x.io; Ada.Lovelace@example.org again."
    )
    assert text == "Write to [EMAIL_1] or [EMAIL_2]; [EMAIL_1] again." and found == {"email": 3}


@pytest.mark.parametrize(
    "number",
    ["4111 1111 1111 1111", "4111-1111-1111-1111", "4111111111111111", "5500 0000 0000 0004"],
)
def test_valid_card_numbers_are_redacted(number):
    assert redact(f"Card: {number}.")[0] == "Card: [CARD_1]."


@pytest.mark.parametrize(
    "number",
    ["4111 1111 1111 1112", "1234 5678 9012 3456", "0000 0000 0000 0000", "111111111111111"],
)
def test_numbers_that_fail_the_luhn_check_or_repeat_one_digit_are_left_alone(number):
    assert "[CARD" not in redact(f"Order {number}.")[0]


def test_iban_and_checksums():
    assert iban_valid("GB82 WEST 1234 5698 7654 32") and not iban_valid(
        "GB82 WEST 1234 5698 7654 33"
    )
    assert redact("Pay to GB82 WEST 1234 5698 7654 32 today")[0] == "Pay to [IBAN_1] today"
    assert redact("Pay to GB82 WEST 1234 5698 7654 33 today")[0].count("[IBAN") == 0
    assert luhn_valid("4111111111111111") and not luhn_valid("4111111111111112")


def test_phone_numbers_in_common_formats():
    for number in ("+44 20 7946 0958", "(415) 555-2671", "+1-415-555-2671", "020 7946 0958"):
        assert redact(f"Call {number} now")[0] == "Call [PHONE_1] now", number


def test_dates_versions_and_short_numbers_are_not_phones():
    for text in (
        "on 2024-01-15 we shipped",
        "version 3.14.15",
        "ticket 12345",
        "page 12-14",
        "in 2019-2024",
    ):
        assert "[PHONE" not in redact(text)[0], text


def test_ssn_and_ip_addresses():
    assert redact("SSN 123-45-6789 on file")[0] == "SSN [SSN_1] on file"
    assert redact("SSN 000-12-3456 and 666-12-3456 and 900-12-3456")[0].count("[SSN") == 0
    assert redact("Host 192.168.1.20 down")[0] == "Host [IP_1] down"
    assert "[IP" not in redact("version 999.1.2.3 released")[0]


def test_only_the_requested_kinds_are_redacted():
    text, found = redact("ada@x.org +44 20 7946 0958", kinds=("email",))
    assert text == "[EMAIL_1] +44 20 7946 0958" and found == {"email": 1}


def test_unknown_kinds_are_rejected():
    with pytest.raises(ValueError, match="unknown PII kinds"):
        Redactor(("email", "passport"))


def test_overlapping_matches_keep_the_leftmost_longest():
    # a card number contains digit runs a phone pattern also matches; the card wins
    assert redact("4111 1111 1111 1111")[0] == "[CARD_1]"


def test_text_without_personal_data_is_unchanged():
    text = "Refunds are issued within 14 days of purchase. See section 3.2 on page 14."
    assert redact(text) == (text, {})


def test_placeholders_stay_consistent_across_calls_on_one_redactor():
    r = Redactor()
    assert (
        r.redact("ada@x.org") == "[EMAIL_1]"
        and r.redact("bob@x.org, ada@x.org") == "[EMAIL_2], [EMAIL_1]"
    )
    assert r.found == {"email": 3}


def test_a_new_redactor_starts_numbering_again():
    assert Redactor().redact("a@x.org") == "[EMAIL_1]" == Redactor().redact("b@y.org")
