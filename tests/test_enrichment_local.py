import pytest

from app.ingest.enrichment.keywords import extract_keywords
from app.ingest.enrichment.language import STOPWORDS, detect_language, dominant_language

SAMPLES = {
    "en": "The company announced that it will open a new office in the city next year, and the "
    "staff are very happy about the plans for this move.",
    "es": "La empresa anunció que abrirá una nueva oficina en la ciudad el próximo año, y los "
    "empleados están muy contentos con los planes para este traslado.",
    "fr": "La société a annoncé qu'elle ouvrira un nouveau bureau dans la ville l'année "
    "prochaine, et les employés sont très contents de ce déménagement.",
    "de": "Das Unternehmen hat angekündigt, dass es im nächsten Jahr ein neues Büro in der Stadt "
    "eröffnen wird, und die Mitarbeiter sind mit den Plänen sehr zufrieden.",
    "it": "L'azienda ha annunciato che aprirà un nuovo ufficio nella città il prossimo anno, e i "
    "dipendenti sono molto contenti dei piani per questo trasferimento.",
    "pt": "A empresa anunciou que vai abrir um novo escritório na cidade no próximo ano, e os "
    "funcionários estão muito felizes com os planos para esta mudança.",
    "nl": "Het bedrijf heeft aangekondigd dat het volgend jaar een nieuw kantoor in de stad zal "
    "openen, en de medewerkers zijn erg blij met de plannen voor deze verhuizing.",
    "sv": "Företaget meddelade att det kommer att öppna ett nytt kontor i staden nästa år, och "
    "de anställda är mycket nöjda med planerna för den här flytten.",
}


@pytest.mark.parametrize("code", list(SAMPLES))
def test_detects_latin_script_languages(code):
    guess = detect_language(SAMPLES[code])
    assert guess is not None and guess.code == code
    assert 0.5 <= guess.confidence <= 1.0


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("これは日本語のテキストです。検索のためのテストをしています。", "ja"),
        ("这是一个用于搜索的中文测试文本，我们正在检查语言检测功能。", "zh"),
        ("이것은 검색을 위한 한국어 테스트 텍스트입니다. 언어 감지를 확인합니다.", "ko"),
        ("Это тестовый текст на русском языке для проверки определения языка.", "ru"),
        ("هذا نص تجريبي باللغة العربية لاختبار اكتشاف اللغة في النظام.", "ar"),
        ("זהו טקסט בדיקה בעברית כדי לבדוק את זיהוי השפה במערכת שלנו.", "he"),
        ("यह भाषा पहचान की जांच के लिए हिंदी में एक परीक्षण पाठ है।", "hi"),
        ("นี่คือข้อความทดสอบภาษาไทยสำหรับตรวจสอบการตรวจจับภาษาของระบบ", "th"),
        ("Αυτό είναι ένα δοκιμαστικό κείμενο στα ελληνικά για έλεγχο γλώσσας.", "el"),
    ],
)
def test_detects_other_scripts(text, code):
    assert detect_language(text).code == code


@pytest.mark.parametrize(
    "text",
    ["", "ok", "12345 67890 11111 22222", "x" * 50, "Zorblat quux fnord blorp wibble wobble"],
)
def test_too_short_or_unrecognisable_text_gives_no_guess(text):
    assert detect_language(text) is None


def test_mixed_text_follows_the_majority_language():
    text = SAMPLES["de"] + " " + SAMPLES["de"] + " API gateway"
    assert detect_language(text).code == "de"


def test_dominant_language():
    assert dominant_language(["en", "de", "en", None]) == "en"
    assert dominant_language([None, None]) is None
    assert dominant_language([]) is None
    assert dominant_language(["de", "en"]) == "en"  # ties are resolved deterministically


def test_stopword_lists_are_sane():
    assert all(len(words) >= 25 for words in STOPWORDS.values())


# --- keywords -------------------------------------------------------------------------------

CHUNKS = [
    "The refund policy allows customers to request a refund within 14 days. The refund policy "
    "applies to all subscriptions, and each refund is processed by the billing team.",
    "Shipping times depend on the destination. International shipping takes 7 to 14 days, and "
    "tracking numbers are emailed once the parcel has shipped.",
    "Our support team answers questions about invoices, billing and subscriptions every weekday.",
]


def test_keywords_are_specific_to_each_chunk():
    kws = extract_keywords(CHUNKS, k=5)
    assert len(kws) == 3
    assert "refund policy" in kws[0] or "refund" in kws[0]
    assert any("shipping" in k for k in kws[1])
    assert any(k in ("support", "support team", "invoices", "weekday") for k in kws[2])
    # a word that is specific to one chunk should not top another chunk's list
    assert "shipping" not in " ".join(kws[0])


def test_repeated_phrases_beat_their_parts():
    kws = extract_keywords(CHUNKS, k=4)[0]
    assert "refund policy" in kws
    assert "policy" not in kws  # covered by the phrase


def test_stopwords_numbers_and_short_words_are_never_keywords():
    kws = extract_keywords(["The and of 12345 is it to a we be of the and"], k=8)[0]
    assert kws == []


def test_k_limits_the_list_and_output_is_deterministic():
    a = extract_keywords(CHUNKS, k=2)
    b = extract_keywords(CHUNKS, k=2)
    assert a == b and all(len(x) <= 2 for x in a)


def test_cjk_text_yields_no_keywords_instead_of_garbage():
    assert extract_keywords(["これは日本語のテキストです。" * 5], k=5) == [[]]


def test_works_with_non_english_text_and_single_chunks():
    kws = extract_keywords([SAMPLES["de"]], k=4)[0]
    assert kws and not {"das", "und", "der"} & set(kws)
    assert extract_keywords([], k=3) == []
