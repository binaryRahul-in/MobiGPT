#!/usr/bin/env node
/**
 * Runs the app's TypeScript G2P (compiled to build/tts-js) on text and prints
 * Kokoro token windows for `mobigpt-rvc tts`. Used by CI and for local checks.
 *
 *   npx tsc -p tools/tts/tsconfig.json
 *   node tools/tts/phonemize.js --lexicon-dir DIR "Some text."   -> JSON {phonemes, tokens, pauses, unknown}
 */
const fs = require('fs');
const path = require('path');
const {EnglishG2P, toWindows} = require(path.join(__dirname, '../../build/tts-js/g2p.js'));

const args = process.argv.slice(2);
const dirIdx = args.indexOf('--lexicon-dir');
const dir = dirIdx >= 0 ? args.splice(dirIdx, 2)[1] : '.';
const text = args.join(' ');
const load = name => JSON.parse(fs.readFileSync(path.join(dir, name), 'utf8'));
const g2p = new EnglishG2P(load('us_gold.json'), load('us_silver.json'));
const {phonemes, unknown} = g2p.phonemize(text);
const windows = toWindows(phonemes);
process.stdout.write(
  JSON.stringify({
    phonemes,
    unknown,
    tokens: windows.map(w => w.tokens.join(',')).join(';'),
    pauses: windows.map(w => w.pauseAfter).join(';'),
  }) + '\n',
);
