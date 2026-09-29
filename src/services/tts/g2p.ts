/**
 * English grapheme -> phoneme conversion for Kokoro, ported from the lexicon path
 * of misaki (hexgrad/misaki, Apache-2.0): gold + silver US dictionaries, misaki's
 * -s / -ed / -ing morphology, numbers read as words, acronyms and unknown words
 * spelled out. misaki's spaCy part-of-speech disambiguation is not ported; words
 * with several readings use the lexicon's DEFAULT entry.
 */
import {KOKORO_VOCAB} from './vocab';

export type LexiconEntry = string | {[tag: string]: string | null};
export type Lexicon = Record<string, LexiconEntry>;

/** Words the upstream lexicons don't know but this app says a lot. */
export const BUILTIN_LEXICON: Lexicon = {
  MobiGPT: 'mˈObi ʤˌi pˌi tˈi',
  Kokoro: 'kˈOkəɹˌO',
  RVC: 'ˌɑɹ vˌi sˈi',
};

/** Kokoro takes at most 510 tokens per inference (512 with the two pads). */
export const MAX_WINDOW_TOKENS = 510;

const US_TAUS = new Set('AIOWYiuæɑəɛɪɹʊʌ');
const PUNCT = new Set([';', ':', ',', '.', '!', '?', '—', '…', '"', '(', ')', '“', '”']);
const OPENERS = new Set(['(', '“']);

export interface Phonemized {
  phonemes: string;
  /** Words that were not in the lexicon and had to be spelled out. */
  unknown: string[];
}

export interface Window {
  tokens: number[];
  phonemes: string;
  /** Seconds of silence to insert after this window. */
  pauseAfter: number;
}

export class EnglishG2P {
  constructor(private gold: Lexicon, private silver: Lexicon = {}, private extra: Lexicon = BUILTIN_LEXICON) {}

  phonemize(text: string): Phonemized {
    const unknown: string[] = [];
    let out = '';
    let attachNext = true; // the next word joins without a space (start, or after an opener)
    let inQuote = false;
    for (const tok of tokenize(normalize(text))) {
      if (PUNCT.has(tok)) {
        const opens = OPENERS.has(tok) || (tok === '"' && !inQuote);
        if (tok === '"') {
          inQuote = !inQuote;
        }
        if (opens) {
          out += attachNext ? tok : ` ${tok}`;
          attachNext = true;
        } else {
          out += tok; // closers and clause punctuation hug the previous word
        }
        continue;
      }
      const words = /^\d/.test(tok) ? numberToWords(tok).split(' ') : [tok];
      for (const w of words) {
        const ps = this.word(w, unknown);
        if (ps) {
          out += attachNext ? ps : ` ${ps}`;
          attachNext = false;
        }
      }
    }
    return {phonemes: out.trim(), unknown};
  }

  /** Phonemes for one word (no spaces inside unless it had to be split). */
  word(raw: string, unknown: string[] = []): string {
    const word = raw.replace(/[‘’]/g, "'");
    const direct = this.lookup(word);
    if (direct) {
      return direct;
    }
    // Hyphenated or camel-cased compounds: read the parts.
    const parts = word.split(/-+/).flatMap(p => p.match(/[A-Z]{2,}(?=[A-Z][a-z]|\b|$)|[A-Z]?[a-z']+|[A-Z]+|\d+/g) ?? []);
    if (parts.length > 1) {
      return parts.map(p => this.word(p, unknown)).join(' ');
    }
    unknown.push(word);
    return this.spell(word);
  }

  private get(word: string): string | null {
    const e = this.extra[word] ?? this.gold[word] ?? this.silver[word];
    if (e == null) {
      return null;
    }
    if (typeof e === 'string') {
      return e;
    }
    return e.DEFAULT ?? Object.values(e).find((v): v is string => typeof v === 'string') ?? null;
  }

  private known(word: string): boolean {
    return this.get(word) != null;
  }

  private lookup(word: string): string | null {
    const lower = word.toLowerCase();
    const candidates = [word];
    if (word !== lower) {
      candidates.push(lower, word[0] + word.slice(1).toLowerCase());
    }
    for (const c of candidates) {
      const p = this.get(c);
      if (p) {
        return p;
      }
    }
    return this.stemS(lower) ?? this.stemEd(lower) ?? this.stemIng(lower);
  }

  // misaki: -s / -es / -ies / 's
  private stemS(word: string): string | null {
    if (word.length < 3 || !word.endsWith('s')) {
      return null;
    }
    let stem: string | null = null;
    if (!word.endsWith('ss') && this.known(word.slice(0, -1))) {
      stem = word.slice(0, -1);
    } else if (
      (word.endsWith("'s") || (word.length > 4 && word.endsWith('es') && !word.endsWith('ies'))) &&
      this.known(word.slice(0, -2))
    ) {
      stem = word.slice(0, -2);
    } else if (word.length > 4 && word.endsWith('ies') && this.known(word.slice(0, -3) + 'y')) {
      stem = word.slice(0, -3) + 'y';
    }
    const ps = stem ? this.get(stem) : null;
    if (!ps) {
      return null;
    }
    const last = ps[ps.length - 1];
    if ('ptkfθ'.includes(last)) {
      return ps + 's';
    }
    if ('szʃʒʧʤ'.includes(last)) {
      return ps + 'ᵻz';
    }
    return ps + 'z';
  }

  // misaki: -d / -ed
  private stemEd(word: string): string | null {
    if (word.length < 4 || !word.endsWith('d')) {
      return null;
    }
    let stem: string | null = null;
    if (!word.endsWith('dd') && this.known(word.slice(0, -1))) {
      stem = word.slice(0, -1);
    } else if (word.length > 4 && word.endsWith('ed') && !word.endsWith('eed') && this.known(word.slice(0, -2))) {
      stem = word.slice(0, -2);
    }
    const ps = stem ? this.get(stem) : null;
    if (!ps) {
      return null;
    }
    const last = ps[ps.length - 1];
    if ('pkfθʃsʧ'.includes(last)) {
      return ps + 't';
    }
    if (last === 'd') {
      return ps + 'ᵻd';
    }
    if (last !== 't') {
      return ps + 'd';
    }
    if (ps.length >= 2 && US_TAUS.has(ps[ps.length - 2])) {
      return ps.slice(0, -1) + 'ɾᵻd';
    }
    return ps + 'ᵻd';
  }

  // misaki: -ing
  private stemIng(word: string): string | null {
    if (word.length < 5 || !word.endsWith('ing')) {
      return null;
    }
    const base = word.slice(0, -3);
    let stem: string | null = null;
    if (word.length > 5 && this.known(base)) {
      stem = base;
    } else if (this.known(base + 'e')) {
      stem = base + 'e';
    } else if (word.length > 5 && /([bcdgklmnprstvxz])\1ing$|cking$/.test(word) && this.known(word.slice(0, -4))) {
      stem = word.slice(0, -4);
    }
    const ps = stem ? this.get(stem) : null;
    if (!ps) {
      return null;
    }
    if (ps.length > 1 && ps.endsWith('t') && US_TAUS.has(ps[ps.length - 2])) {
      return ps.slice(0, -1) + 'ɾɪŋ';
    }
    return ps + 'ɪŋ';
  }

  /** Letter-by-letter, e.g. "GPT" -> "ʤˈi pˈi tˈi". */
  private spell(word: string): string {
    return [...word.toUpperCase()]
      .map(ch => this.get(ch) ?? (/\d/.test(ch) ? this.lookup(numberToWords(ch)) : null))
      .filter((p): p is string => !!p)
      .join(' ');
  }
}

// ------------------------------------------------------------------ text

export function normalize(text: string): string {
  return text
    .normalize('NFKC')
    .replace(/[‘’`´]/g, "'")
    .replace(/\r?\n+/g, '. ')
    .replace(/&/g, ' and ')
    .replace(/(\d)\s*%/g, '$1 percent')
    .replace(/\$(\d+(?:[.,]\d+)*)/g, '$1 dollars')
    .replace(/(^|\s)-(?=\s)|\s--?\s|–/g, ' — ')
    .replace(/\.{3,}/g, '…')
    .replace(/[\t ]+/g, ' ')
    .trim();
}

export function tokenize(text: string): string[] {
  return text.match(/\d+(?:[.,]\d+)*(?:st|nd|rd|th)?|[A-Za-z]+(?:['-][A-Za-z]+)*'?|[;:,.!?—…"()“”]/g) ?? [];
}

// ---------------------------------------------------------------- numbers

const ONES = [
  'zero',
  'one',
  'two',
  'three',
  'four',
  'five',
  'six',
  'seven',
  'eight',
  'nine',
  'ten',
  'eleven',
  'twelve',
  'thirteen',
  'fourteen',
  'fifteen',
  'sixteen',
  'seventeen',
  'eighteen',
  'nineteen',
];
const TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];
const SCALES: [number, string][] = [
  [1e12, 'trillion'],
  [1e9, 'billion'],
  [1e6, 'million'],
  [1e3, 'thousand'],
];
const ORDINAL_IRREGULAR: Record<string, string> = {
  one: 'first',
  two: 'second',
  three: 'third',
  five: 'fifth',
  eight: 'eighth',
  nine: 'ninth',
  twelve: 'twelfth',
};

function below1000(n: number): string {
  const out: string[] = [];
  if (n >= 100) {
    out.push(ONES[Math.floor(n / 100)], 'hundred');
    n %= 100;
  }
  if (n >= 20) {
    out.push(TENS[Math.floor(n / 10)] + (n % 10 ? `-${ONES[n % 10]}` : ''));
  } else if (n > 0 || out.length === 0) {
    out.push(ONES[n]);
  }
  return out.join(' ');
}

export function cardinal(n: number): string {
  if (!Number.isFinite(n) || n < 0 || n >= 1e15) {
    return String(n)
      .split('')
      .map(d => ONES[Number(d)] ?? '')
      .join(' ');
  }
  if (n < 1000) {
    return below1000(n);
  }
  const out: string[] = [];
  for (const [value, name] of SCALES) {
    if (n >= value) {
      out.push(below1000(Math.floor(n / value)), name);
      n %= value;
    }
  }
  if (n > 0) {
    out.push(below1000(n));
  }
  return out.join(' ');
}

function ordinal(n: number): string {
  const words = cardinal(n).split(' ');
  const last = words.pop()!;
  const [head, tail] = last.includes('-') ? [last.slice(0, last.lastIndexOf('-') + 1), last.slice(last.lastIndexOf('-') + 1)] : ['', last];
  const ord = ORDINAL_IRREGULAR[tail] ?? (tail.endsWith('y') ? `${tail.slice(0, -1)}ieth` : `${tail}th`);
  return [...words, head + ord].join(' ');
}

/** Reads a numeric token as English words (hyphens become spaces for lexicon lookup). */
export function numberToWords(token: string): string {
  const m = /^(\d+(?:[.,]\d+)*)(st|nd|rd|th)?$/.exec(token);
  if (!m) {
    return token;
  }
  const [, num, suffix] = m;
  let words: string;
  if (suffix) {
    words = ordinal(Number(num.replace(/,/g, '')));
  } else if (/^\d{1,3}(,\d{3})+$/.test(num)) {
    words = cardinal(Number(num.replace(/,/g, '')));
  } else if (/^\d+[.,]\d+$/.test(num)) {
    const [int, frac] = num.split(/[.,]/);
    words = `${cardinal(Number(int))} point ${frac
      .split('')
      .map(d => ONES[Number(d)])
      .join(' ')}`;
  } else if (/^\d{4}$/.test(num) && Number(num) >= 1100 && Number(num) <= 2099 && Number(num) % 1000 >= 10) {
    // Years: 1999 -> nineteen ninety-nine, 2025 -> twenty twenty-five, 1905 -> nineteen oh five.
    const hi = Number(num.slice(0, 2));
    const lo = Number(num.slice(2));
    words = `${below1000(hi)} ${lo === 0 ? 'hundred' : lo < 10 ? `oh ${ONES[lo]}` : below1000(lo)}`;
  } else if (/^0\d+$/.test(num)) {
    words = num
      .split('')
      .map(d => ONES[Number(d)])
      .join(' ');
  } else {
    words = cardinal(Number(num));
  }
  return words.replace(/-/g, ' ');
}

// ---------------------------------------------------------------- windows

export function toTokens(phonemes: string): number[] {
  const ids: number[] = [];
  for (const ch of phonemes) {
    const id = KOKORO_VOCAB[ch];
    if (id != null) {
      ids.push(id);
    }
  }
  return ids;
}

/**
 * Splits a phoneme string into model windows of at most `max` tokens, preferring
 * sentence ends, then clause punctuation, then word boundaries.
 */
export function toWindows(phonemes: string, max = MAX_WINDOW_TOKENS): Window[] {
  const out: Window[] = [];
  const sentences = phonemes.match(/[^.!?…]+[.!?…]*\s*/g) ?? [];
  let current = '';
  const flush = (pauseAfter: number) => {
    const p = current.trim();
    if (p) {
      out.push({phonemes: p, tokens: toTokens(p), pauseAfter});
    }
    current = '';
  };
  for (const sentence of sentences) {
    if (toTokens(current + sentence).length <= max) {
      current += sentence;
      continue;
    }
    flush(0.05);
    if (toTokens(sentence).length <= max) {
      current = sentence;
      continue;
    }
    // A single sentence longer than a window: break at clauses, then words.
    for (const piece of sentence.split(/(?<=[,;:—])\s+|\s+/)) {
      if (toTokens(`${current} ${piece}`).length > max) {
        flush(0.02);
      }
      current = current ? `${current} ${piece}` : piece;
    }
  }
  flush(0);
  return out;
}
