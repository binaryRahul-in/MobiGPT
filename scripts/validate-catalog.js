#!/usr/bin/env node
/**
 * Verifies every file referenced by catalog/models.json and catalog/voices.json
 * exists on the Hugging Face Hub (and that its size is close to the catalog's
 * estimate). Run in CI so broken presets are caught before users hit them.
 *   node scripts/validate-catalog.js [--fix]   (--fix rewrites sizes in place)
 */
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
const fix = process.argv.includes('--fix');
const headers = process.env.HF_TOKEN ? {Authorization: `Bearer ${process.env.HF_TOKEN}`} : {};

async function listing(repo) {
  const res = await fetch(`https://huggingface.co/api/models/${repo}/tree/main?recursive=true`, {headers});
  if (!res.ok) {
    throw new Error(`${repo}: HTTP ${res.status}`);
  }
  return (await res.json()).filter(e => e.type === 'file').map(e => ({path: e.path, size: e.lfs?.size ?? e.size}));
}

async function main() {
  const models = JSON.parse(fs.readFileSync(path.join(root, 'catalog/models.json'), 'utf8'));
  const voices = JSON.parse(fs.readFileSync(path.join(root, 'catalog/voices.json'), 'utf8'));
  const entries = [
    ...models.models.map(m => ({group: 'models', e: m})),
    ...voices.encoders.map(m => ({group: 'encoders', e: m})),
    ...voices.pitch.map(m => ({group: 'pitch', e: m})),
    ...voices.voices.map(m => ({group: 'voices', e: m})),
    // Neural TTS: the Hugging Face copies are the fallback behind the release mirror.
    ...(voices.tts?.models ?? []).map(m => ({group: 'tts', e: m})),
    ...(voices.tts?.voices ?? []).map(m => ({group: 'tts-voices', e: m})),
  ];
  const cache = new Map();
  let failures = 0;
  for (const {group, e} of entries) {
    try {
      if (!cache.has(e.repo)) {
        cache.set(e.repo, await listing(e.repo));
      }
      const files = cache.get(e.repo);
      const f = files.find(x => x.path === e.file) ?? files.find(x => x.path.toLowerCase() === e.file.toLowerCase());
      if (!f) {
        failures++;
        const similar = files.filter(x => /\.(gguf|onnx)$/i.test(x.path)).slice(0, 6).map(x => x.path);
        console.log(`✗ ${group}/${e.id}: ${e.file} not in ${e.repo}. Available: ${similar.join(', ')}`);
        continue;
      }
      const drift = e.sizeBytes ? Math.abs(f.size - e.sizeBytes) / f.size : 0;
      console.log(`✓ ${group}/${e.id}: ${f.path} ${(f.size / 1e6).toFixed(1)} MB${drift > 0.05 ? ` (catalog says ${(e.sizeBytes / 1e6).toFixed(1)} MB)` : ''}`);
      if (fix && e.sizeBytes !== undefined) {
        e.sizeBytes = f.size;
        e.file = f.path;
      }
    } catch (err) {
      failures++;
      console.log(`✗ ${group}/${e.id}: ${err.message}`);
    }
  }
  if (fix) {
    fs.writeFileSync(path.join(root, 'catalog/models.json'), JSON.stringify(models, null, 2) + '\n');
    fs.writeFileSync(path.join(root, 'catalog/voices.json'), JSON.stringify(voices, null, 2) + '\n');
  }
  console.log(`\n${entries.length - failures}/${entries.length} catalog entries resolved`);
  process.exit(failures ? 1 : 0);
}

main();
