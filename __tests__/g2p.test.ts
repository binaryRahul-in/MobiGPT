import {EnglishG2P, Lexicon, MAX_WINDOW_TOKENS, numberToWords, toTokens, toWindows} from '../src/services/tts/g2p';
import {KOKORO_VOCAB} from '../src/services/tts/vocab';

// A few dozen entries taken verbatim from misaki's us_gold/us_silver (Apache-2.0).
const lexicon: Lexicon = require('../jest/fixtures-lexicon.json');
const g2p = new EnglishG2P(lexicon);

describe('EnglishG2P', () => {
  it('reads a sentence the way misaki does (lexicon path)', () => {
    expect(g2p.phonemize('Hello world, this voice was made on my phone.').phonemes).toBe('həlˈO wˈɜɹld, ðɪs vˈYs wʌz mˌAd ˌɔn mI fˈOn.');
  });

  it('uses the DEFAULT reading for words with several parts of speech', () => {
    expect(g2p.word('read')).toBe('ɹˈid');
    expect(g2p.word('convert')).toBe('kənvˈɜɹt');
  });

  it('applies misaki -s / -ed / -ing morphology to known stems', () => {
    expect(g2p.word('voices')).toBe('vˈYsᵻz');
    expect(g2p.word('plays')).toBe('plˈAz');
    expect(g2p.word('played')).toBe('plˈAd');
    expect(g2p.word('converted')).toBe('kənvˈɜɹɾᵻd');
    expect(g2p.word('playing')).toBe('plˈAɪŋ');
    expect(g2p.word('getting')).toBe('ɡɛɾɪŋ');
    expect(g2p.word("world's")).toBe('wˈɜɹldz');
  });

  it('matches capitalised words and spells unknown acronyms', () => {
    expect(g2p.word('Hello')).toBe('həlˈO');
    const r = g2p.phonemize('GPT');
    expect(r.phonemes).toBe('ʤˈi pˈi tˈi');
    expect(r.unknown).toEqual(['GPT']);
  });

  it('knows the app\'s own name', () => {
    expect(g2p.phonemize('MobiGPT').unknown).toEqual([]);
    expect(g2p.word('MobiGPT')).toBe('mˈObi ʤˌi pˌi tˈi');
  });

  it('splits camel-case and hyphenated compounds', () => {
    expect(g2p.word('DeviceGPT', [])).toBe(`${g2p.word('Device', [])} ʤˈi pˈi tˈi`);
    expect(g2p.word('state-of-the-art')).toBe(['state', 'of', 'the', 'art'].map(w => g2p.word(w)).join(' '));
  });

  it('attaches punctuation and quotes to the right words', () => {
    expect(g2p.phonemize('Hello, "world" (play).').phonemes).toBe('həlˈO, "wˈɜɹld" (plˈA).');
  });

  it('only emits symbols that exist in the Kokoro vocabulary', () => {
    const {phonemes} = g2p.phonemize('Hello world! In 1999 I had 3.5 apples, 21 pears and 50% of $5.');
    expect([...phonemes].filter(ch => KOKORO_VOCAB[ch] == null)).toEqual([]);
  });
});

describe('numbers', () => {
  it.each([
    ['7', 'seven'],
    ['21', 'twenty one'],
    ['105', 'one hundred five'],
    ['1,250', 'one thousand two hundred fifty'],
    ['3.5', 'three point five'],
    ['1999', 'nineteen ninety nine'],
    ['2025', 'twenty twenty five'],
    ['1905', 'nineteen oh five'],
    ['2000', 'two thousand'],
    ['1st', 'first'],
    ['22nd', 'twenty second'],
    ['40th', 'fortieth'],
    ['007', 'zero zero seven'],
  ])('%s -> %s', (n, words) => expect(numberToWords(n)).toBe(words));
});

describe('windows', () => {
  it('keeps short text in one window and maps symbols to token ids', () => {
    const w = toWindows('həlˈO wˈɜɹld.');
    expect(w).toHaveLength(1);
    expect(w[0].tokens).toEqual(toTokens('həlˈO wˈɜɹld.'));
    expect(w[0].tokens[0]).toBe(KOKORO_VOCAB.h);
  });

  it('never exceeds the model limit and prefers sentence boundaries', () => {
    const sentence = 'həlˈO wˈɜɹld, ðɪs vˈYs wʌz mˌAd ˌɔn mI fˈOn. ';
    const text = sentence.repeat(40);
    const windows = toWindows(text);
    expect(windows.length).toBeGreaterThan(1);
    for (const w of windows) {
      expect(w.tokens.length).toBeLessThanOrEqual(MAX_WINDOW_TOKENS);
      expect(w.phonemes.endsWith('.')).toBe(true);
    }
    expect(windows.map(w => w.tokens.length).reduce((a, b) => a + b, 0)).toBeGreaterThan(toTokens(text.trim()).length - 40);
  });

  it('splits a single overlong sentence at word boundaries', () => {
    const long = Array.from({length: 200}, () => 'wˈɜɹld').join(' ') + '.';
    const windows = toWindows(long);
    expect(windows.length).toBeGreaterThan(1);
    windows.forEach(w => expect(w.tokens.length).toBeLessThanOrEqual(MAX_WINDOW_TOKENS));
  });
});
