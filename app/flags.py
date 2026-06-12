"""
Flag emojis for national teams.

``flag(name)`` returns the country's flag emoji (built from its ISO 3166-1 alpha-2
code via regional-indicator symbols), or an empty string for unknown names.
England, Scotland and Wales use their subdivision flag sequences. Names follow the
martj42 international-results spelling. Club teams (leagues) have no flag.
"""
from __future__ import annotations

__all__ = ["flag", "with_flag"]

# Subdivision flags (tag sequences) — not expressible as ISO-2 regional indicators.
_SPECIAL: dict[str, str] = {
    "England": "\U0001F3F4\U000E0067\U000E0062\U000E0065\U000E006E\U000E0067\U000E007F",
    "Scotland": "\U0001F3F4\U000E0067\U000E0062\U000E0073\U000E0063\U000E0074\U000E007F",
    "Wales": "\U0001F3F4\U000E0067\U000E0062\U000E0077\U000E006C\U000E0073\U000E007F",
}

# Country name (martj42 spelling) -> ISO 3166-1 alpha-2 code.
_ISO2: dict[str, str] = {
    # 2026 World Cup teams
    "Mexico": "MX", "Canada": "CA", "Brazil": "BR", "United States": "US",
    "Germany": "DE", "Netherlands": "NL", "Belgium": "BE", "Spain": "ES",
    "France": "FR", "Argentina": "AR", "Portugal": "PT", "Czech Republic": "CZ",
    "South Africa": "ZA", "South Korea": "KR", "Bosnia and Herzegovina": "BA",
    "Qatar": "QA", "Switzerland": "CH", "Haiti": "HT", "Morocco": "MA",
    "Australia": "AU", "Paraguay": "PY", "Turkey": "TR", "Curaçao": "CW",
    "Curacao": "CW", "Ecuador": "EC", "Ivory Coast": "CI", "Côte d'Ivoire": "CI",
    "Japan": "JP", "Sweden": "SE", "Tunisia": "TN", "Egypt": "EG", "Iran": "IR",
    "New Zealand": "NZ", "Cape Verde": "CV", "Saudi Arabia": "SA", "Uruguay": "UY",
    "Iraq": "IQ", "Norway": "NO", "Senegal": "SN", "Algeria": "DZ", "Austria": "AT",
    "Jordan": "JO", "Colombia": "CO", "DR Congo": "CD", "Uzbekistan": "UZ",
    "Croatia": "HR", "Ghana": "GH", "Panama": "PA",
    # other prominent national teams (ratings / predictor)
    "Italy": "IT", "Nigeria": "NG", "Cameroon": "CM", "Chile": "CL", "Peru": "PE",
    "Venezuela": "VE", "Bolivia": "BO", "Costa Rica": "CR", "Honduras": "HN",
    "Jamaica": "JM", "Denmark": "DK", "Poland": "PL", "Serbia": "RS", "Ukraine": "UA",
    "Greece": "GR", "Hungary": "HU", "Romania": "RO", "Russia": "RU",
    "Republic of Ireland": "IE", "Ireland": "IE", "Northern Ireland": "GB",
    "Finland": "FI", "Iceland": "IS", "Slovenia": "SI", "Slovakia": "SK",
    "Bulgaria": "BG", "Israel": "IL", "Mali": "ML", "Burkina Faso": "BF",
    "Zambia": "ZM", "Guinea": "GN", "Congo": "CG", "Angola": "AO", "Mozambique": "MZ",
    "Kenya": "KE", "Uganda": "UG", "Tanzania": "TZ", "Zimbabwe": "ZW", "Gabon": "GA",
    "Benin": "BJ", "Mauritania": "MR", "Madagascar": "MG", "Namibia": "NA",
    "Equatorial Guinea": "GQ", "Sudan": "SD", "Libya": "LY", "Togo": "TG",
    "China": "CN", "China PR": "CN", "India": "IN", "Thailand": "TH", "Vietnam": "VN",
    "Indonesia": "ID", "Malaysia": "MY", "Philippines": "PH", "United Arab Emirates": "AE",
    "Bahrain": "BH", "Kuwait": "KW", "Oman": "OM", "Lebanon": "LB", "Syria": "SY",
    "Palestine": "PS", "Kyrgyzstan": "KG", "Tajikistan": "TJ", "Turkmenistan": "TM",
    "North Korea": "KP", "Hong Kong": "HK", "Guatemala": "GT", "El Salvador": "SV",
    "Trinidad and Tobago": "TT", "Albania": "AL", "North Macedonia": "MK",
    "Montenegro": "ME", "Kosovo": "XK", "Georgia": "GE", "Armenia": "AM",
    "Azerbaijan": "AZ", "Kazakhstan": "KZ", "Belarus": "BY", "Estonia": "EE",
    "Latvia": "LV", "Lithuania": "LT", "Luxembourg": "LU", "Malta": "MT",
    "Cyprus": "CY", "Faroe Islands": "FO", "Gibraltar": "GI", "Andorra": "AD",
    "San Marino": "SM", "Liechtenstein": "LI", "Moldova": "MD",
}


def _iso_flag(code: str) -> str:
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in code.upper())


def flag(name: str) -> str:
    """Return the flag emoji for a country name, or ``""`` if unknown."""
    if name in _SPECIAL:
        return _SPECIAL[name]
    code = _ISO2.get(name)
    return _iso_flag(code) if code else ""


def with_flag(name: str, sep: str = " ") -> str:
    """Prefix ``name`` with its flag emoji when available."""
    emoji = flag(name)
    return f"{emoji}{sep}{name}" if emoji else name
