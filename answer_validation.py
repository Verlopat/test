"""Conservative formatting checks for known-answer smoke tests.

This is NOT a semantic or factual correctness judge.
"""
from decimal import Decimal, InvalidOperation
import re

NUMBER = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'


def matches_expected(response: str, expected: str) -> bool:
    """Accept simple final numeric answers even when formatted as a sentence.

    Examples: 156, 156.0, Answer: 156, The answer is **156**.,
    12 * 13 = 156. Do not accept an arbitrary mention of 156 in prose.
    """
    actual = response.strip()
    wanted = expected.strip()
    if actual == wanted:
        return True
    if not re.fullmatch(NUMBER, wanted):
        return False
    if not actual:
        return False

    final = actual.splitlines()[-1].strip()
    final = final.replace('**', '').replace(chr(96), '').strip()
    final = final.rstrip('.!').strip()
    if final.startswith(r'\boxed{') and final.endswith('}'):
        final = final[7:-1]

    number = None
    if re.fullmatch(NUMBER, final):
        number = final
    else:
        endings = (
            rf'(?i)(?:the\s+)?(?:final\s+)?(?:answer|result)\s*(?::|is|=)\s*({NUMBER})\Z',
            rf'(?i)(?:=|equals|is)\s*({NUMBER})\Z',
        )
        for pattern in endings:
            match = re.search(pattern, final)
            if match is not None:
                number = match.group(1)
                break

    if number is None:
        return False
    try:
        return Decimal(number.replace(',', '')) == Decimal(wanted.replace(',', ''))
    except InvalidOperation:
        return False
