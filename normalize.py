"""
Utilities for normalizing and cleaning Wikipedia ('wiki18') text for
Sentence Relevance Estimation (SRE).

Adapted from chemdataextractor.text.normalize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/mcs07/ChemDataExtractor

Copyright 2016 by Matt Swain.
Released under the MIT license.
"""

import re
import unicodedata

#: Control characters.
CONTROLS = {
    '\u0001', '\u0002', '\u0003', '\u0004', '\u0005', '\u0006', '\u0007', '\u0008', '\u000e', '\u000f', '\u0011',
    '\u0012', '\u0013', '\u0014', '\u0015', '\u0016', '\u0017', '\u0018', '\u0019', '\u001a', '\u001b',
}
# There are further control characters, but they are instead replaced with a space by unicode normalization
# '\u0009', '\u000a', '\u000b', '\u000c', '\u000d', '\u001c',  '\u001d', '\u001e', '\u001f'


#: Hyphen and dash characters.
HYPHENS = {
    '-',  # \u002d Hyphen-minus
    '‐',  # \u2010 Hyphen
    '-',  # \u2011 Non-breaking hyphen
    '⁃',  # \u2043 Hyphen bullet
    '‒',  # \u2012 figure dash
    '–',  # \u2013 en dash
    '—',  # \u2014 em dash
    '―',  # \u2015 horizontal bar
}

#: Minus characters.
MINUSES = {
    '-',  # \u002d Hyphen-minus
    '−',  # \u2212 Minus
    '－',  # \uff0d Full-width Hyphen-minus
    '⁻',  # \u207b Superscript minus
}

#: Plus characters.
PLUSES = {
    '+',  # \u002b Plus
    '＋',  # \uff0b Full-width Plus
    '⁺',  # \u207a Superscript plus
}

#: Slash characters.
SLASHES = {
    '/',  # \u002f Solidus
    '⁄',  # \u2044 Fraction slash
    '∕',  # \u2215 Division slash
}

#: Tilde characters.
TILDES = {
    '~',  # \u007e Tilde
    '˜',  # \u02dc Small tilde
    '⁓',  # \u2053 Swung dash
    '∼',  # \u223c Tilde operator
    '∽',  # \u223d Reversed tilde
    '∿',  # \u223f Sine wave
    '〜',  # \u301c Wave dash
    '～',  # \uff5e Full-width tilde
}

#: Apostrophe characters.
APOSTROPHES = {
    "'",  # \u0027
    '’',  # \u2019
    '՚',  # \u055a
    'Ꞌ',  # \ua78b
    'ꞌ',  # \ua78c
    '＇',  # \uff07
}

#: Single quote characters.
SINGLE_QUOTES = {
    "'",  # \u0027
    '‘',  # \u2018
    '’',  # \u2019
    '‚',  # \u201a
    '‛',  # \u201b
}

#: Double quote characters.
DOUBLE_QUOTES = {
    '"',  # \u0022
    '“',  # \u201c
    '”',  # \u201d
    '„',  # \u201e
    '‟',  # \u201f
}

#: Accent characters.
ACCENTS = {
    '`',  # \u0060
    '´',  # \u00b4
}

#: Prime characters.
PRIMES = {
    '′',  # \u2032
    '″',  # \u2033
    '‴',  # \u2034
    '‵',  # \u2035
    '‶',  # \u2036
    '‷',  # \u2037
    '⁗',  # \u2057
}

#: Quote characters, including apostrophes, single quotes, double quotes, accents and primes.
QUOTES = APOSTROPHES | SINGLE_QUOTES | DOUBLE_QUOTES | ACCENTS | PRIMES


# --- Core Unicode + punctuation normalization (stable, model-friendly) ---
def normalize_unicode(text: str) -> str:
    # Normalize to NFKC to fold full-width forms, compatibility characters, etc.
    text = unicodedata.normalize("NFKC", text)

    for control in CONTROLS:
        text = text.replace(control, '')
    text = text.replace('\u000b', ' ').replace('\u000c', ' ').replace(u'\u0085', ' ')

    for hyphen in HYPHENS | MINUSES:
        text = text.replace(hyphen, '-')
    text = text.replace('\u00ad', '')

    for double_quote in DOUBLE_QUOTES:
        text = text.replace(double_quote, '"')  # \u0022
    for single_quote in (SINGLE_QUOTES | APOSTROPHES | ACCENTS):
        text = text.replace(single_quote, "'")  # \u0027
    text = text.replace('′', "'")     # \u2032 prime
    text = text.replace('‵', "'")     # \u2035 reversed prime
    text = text.replace('″', "''")    # \u2033 double prime
    text = text.replace('‶', "''")    # \u2036 reversed double prime
    text = text.replace('‴', "'''")   # \u2034 triple prime
    text = text.replace('‷', "'''")   # \u2037 reversed triple prime
    text = text.replace('⁗', "''''")  # \u2057 quadruple prime

    text = text.replace('…', '...').replace(' . . . ', ' ... ')  # \u2026

    for slash in SLASHES:
        text = text.replace(slash, '/')

    # If you want to fold all tildes, uncomment below.
    # for tilde in TILDES:
    #     text = text.replace(tilde, '~')

    return text


# --- Wikipedia-specific cleanup: remove markup / noise not useful for SRE ---
def clean_wiki_markup(text: str) -> str:
    # Replace wiki links [[target|anchor]] / [[target]] with the visible anchor/content.
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)

    # Remove <ref>...</ref> references and self-closing <ref .../> tags.
    text = re.sub(r'<ref[^>]*>.*?</ref>', ' ', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*/>', ' ', text)

    # Remove generic HTML / XML-like tags.
    text = re.sub(r'<[^>]+>', ' ', text)

    # Drop simple citation brackets like [1], [2,3], [note 4].
    text = re.sub(r'\[[0-9 ,;]+\]', ' ', text)
    text = re.sub(r'\[note [0-9]+\]', ' ', text, flags=re.IGNORECASE)

    # Remove simple template blocks {{...}} (best-effort, non-nested).
    text = re.sub(r'\{\{.*?\}\}', ' ', text, flags=re.DOTALL)

    return text


# --- Whitespace normalization to make model input compact and consistent ---
def normalize_whitespace(text: str) -> str:
    text = text.replace('\u00a0', ' ')  # non-breaking spaces
    text = text.replace('\t', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# --- Single entry point: call this before feeding text to SRE models ---
def normalize(text: str) -> str:
    text = normalize_unicode(text)
    text = clean_wiki_markup(text)
    text = normalize_whitespace(text)
    return text
