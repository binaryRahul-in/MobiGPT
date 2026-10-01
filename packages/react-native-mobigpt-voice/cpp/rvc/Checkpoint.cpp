#include "Checkpoint.h"

#include <zlib.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <variant>

#include "OrtSession.h"  // floatToHalf

namespace mobigpt::rvc {

namespace {

// ------------------------------------------------------------------ helpers

uint16_t rd16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
uint32_t rd32(const uint8_t* p) { return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24); }
uint64_t rd64(const uint8_t* p) { return static_cast<uint64_t>(rd32(p)) | (static_cast<uint64_t>(rd32(p + 4)) << 32); }

std::vector<uint8_t> readRange(const std::string& path, uint64_t offset, uint64_t size) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  f.seekg(static_cast<std::streamoff>(offset));
  std::vector<uint8_t> buf(size);
  f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(size));
  if (static_cast<uint64_t>(f.gcount()) != size) throw std::runtime_error("unexpected end of " + path);
  return buf;
}

uint64_t fileSize(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  return static_cast<uint64_t>(f.tellg());
}

bool endsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && std::equal(suffix.rbegin(), suffix.rend(), s.rbegin(),
                                                 [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

}  // namespace

// ---------------------------------------------------------------------- ZIP

ZipReader::ZipReader(const std::string& path) : path_(path) {
  const uint64_t size = fileSize(path);
  const uint64_t tail = std::min<uint64_t>(size, 65536 + 22);
  auto buf = readRange(path, size - tail, tail);
  // End of central directory record (search backwards, the comment may follow it).
  int64_t eocd = -1;
  for (int64_t i = static_cast<int64_t>(buf.size()) - 22; i >= 0; --i) {
    if (rd32(&buf[static_cast<size_t>(i)]) == 0x06054b50) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw std::runtime_error("not a ZIP file: " + path);
  const uint8_t* e = &buf[static_cast<size_t>(eocd)];
  uint64_t count = rd16(e + 10);
  uint64_t cdSize = rd32(e + 12);
  uint64_t cdOffset = rd32(e + 16);
  if (cdOffset == 0xFFFFFFFFu || count == 0xFFFF) {  // ZIP64: locator sits right before the EOCD
    if (eocd < 20 || rd32(&buf[static_cast<size_t>(eocd - 20)]) != 0x07064b50) throw std::runtime_error("bad ZIP64 locator");
    const uint64_t z64 = rd64(&buf[static_cast<size_t>(eocd - 20) + 8]);
    auto r = readRange(path, z64, 56);
    if (rd32(r.data()) != 0x06064b50) throw std::runtime_error("bad ZIP64 record");
    count = rd64(r.data() + 32);
    cdSize = rd64(r.data() + 40);
    cdOffset = rd64(r.data() + 48);
  }
  auto cd = readRange(path, cdOffset, cdSize);
  size_t p = 0;
  for (uint64_t i = 0; i < count; ++i) {
    if (p + 46 > cd.size() || rd32(&cd[p]) != 0x02014b50) throw std::runtime_error("corrupt ZIP directory");
    Entry en;
    en.method = rd16(&cd[p + 10]);
    en.compressedSize = rd32(&cd[p + 20]);
    en.size = rd32(&cd[p + 24]);
    const uint16_t nameLen = rd16(&cd[p + 28]), extraLen = rd16(&cd[p + 30]), commentLen = rd16(&cd[p + 32]);
    en.localHeaderOffset = rd32(&cd[p + 42]);
    en.name.assign(reinterpret_cast<const char*>(&cd[p + 46]), nameLen);
    // ZIP64 extended information extra field.
    size_t x = p + 46 + nameLen, xe = x + extraLen;
    while (x + 4 <= xe) {
      const uint16_t id = rd16(&cd[x]), len = rd16(&cd[x + 2]);
      if (id == 0x0001) {
        size_t q = x + 4;
        if (en.size == 0xFFFFFFFFu) en.size = rd64(&cd[q]), q += 8;
        if (en.compressedSize == 0xFFFFFFFFu) en.compressedSize = rd64(&cd[q]), q += 8;
        if (en.localHeaderOffset == 0xFFFFFFFFu) en.localHeaderOffset = rd64(&cd[q]);
      }
      x += 4 + len;
    }
    entries_.push_back(std::move(en));
    p += 46 + nameLen + extraLen + commentLen;
  }
}

const ZipReader::Entry* ZipReader::find(const std::string& name) const {
  for (const auto& e : entries_) {
    if (e.name == name) return &e;
  }
  return nullptr;
}

uint64_t ZipReader::dataOffset(const Entry& e) const {
  auto h = readRange(path_, e.localHeaderOffset, 30);
  if (rd32(h.data()) != 0x04034b50) throw std::runtime_error("corrupt ZIP entry " + e.name);
  return e.localHeaderOffset + 30 + rd16(&h[26]) + rd16(&h[28]);
}

namespace {

// Streams an entry's (decompressed) bytes to `sink` in chunks.
template <typename Sink>
void streamEntry(const std::string& path, uint64_t offset, const ZipReader::Entry& e, Sink sink) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  f.seekg(static_cast<std::streamoff>(offset));
  std::vector<uint8_t> in(1 << 20);
  uint64_t left = e.compressedSize;
  if (e.method == 0) {
    while (left > 0) {
      const auto n = static_cast<std::streamsize>(std::min<uint64_t>(left, in.size()));
      f.read(reinterpret_cast<char*>(in.data()), n);
      if (f.gcount() != n) throw std::runtime_error("unexpected end of ZIP entry " + e.name);
      sink(in.data(), static_cast<size_t>(n));
      left -= static_cast<uint64_t>(n);
    }
    return;
  }
  if (e.method != 8) throw std::runtime_error("unsupported ZIP compression method " + std::to_string(e.method) + " for " + e.name);
  z_stream zs{};
  if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) throw std::runtime_error("zlib init failed");
  std::unique_ptr<z_stream, void (*)(z_stream*)> guard(&zs, [](z_stream* s) { inflateEnd(s); });
  std::vector<uint8_t> out(1 << 20);
  int rc = Z_OK;
  while (rc != Z_STREAM_END) {
    if (zs.avail_in == 0) {
      if (left == 0) throw std::runtime_error("truncated deflate stream in " + e.name);
      const auto n = static_cast<std::streamsize>(std::min<uint64_t>(left, in.size()));
      f.read(reinterpret_cast<char*>(in.data()), n);
      if (f.gcount() != n) throw std::runtime_error("unexpected end of ZIP entry " + e.name);
      left -= static_cast<uint64_t>(n);
      zs.next_in = in.data();
      zs.avail_in = static_cast<uInt>(n);
    }
    zs.next_out = out.data();
    zs.avail_out = static_cast<uInt>(out.size());
    rc = inflate(&zs, Z_NO_FLUSH);
    if (rc != Z_OK && rc != Z_STREAM_END) throw std::runtime_error("corrupt deflate data in " + e.name);
    sink(out.data(), out.size() - zs.avail_out);
  }
}

}  // namespace

std::vector<uint8_t> ZipReader::read(const Entry& e) const {
  std::vector<uint8_t> data;
  data.reserve(e.size);
  streamEntry(path_, dataOffset(e), e, [&](const uint8_t* p, size_t n) { data.insert(data.end(), p, p + n); });
  if (data.size() != e.size) throw std::runtime_error("size mismatch in ZIP entry " + e.name);
  return data;
}

void ZipReader::extract(const Entry& e, const std::string& destPath) const {
  std::ofstream out(destPath, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write " + destPath);
  uint64_t written = 0;
  streamEntry(path_, dataOffset(e), e, [&](const uint8_t* p, size_t n) {
    out.write(reinterpret_cast<const char*>(p), static_cast<std::streamsize>(n));
    written += n;
  });
  if (written != e.size) throw std::runtime_error("size mismatch extracting " + e.name);
}

// ------------------------------------------------------------------- pickle

namespace {

// Just enough of the pickle VM for torch.save() checkpoints. Callables are never
// executed: GLOBALs become names, and REDUCE only recognises tensor rebuilds and
// OrderedDict; anything else becomes an opaque value.
struct PVal;
using PList = std::vector<std::shared_ptr<PVal>>;
struct PTensor {
  std::string storageKey, dtype;
  int64_t offset = 0;
  std::vector<int64_t> shape, stride;
};
struct PStorage {
  std::string key, dtype;
};
struct PGlobal {
  std::string module, name;
};
struct PDict {
  std::vector<std::pair<std::shared_ptr<PVal>, std::shared_ptr<PVal>>> items;
};
struct PNone {};
struct PMark {};
struct PVal {
  std::variant<PNone, PMark, bool, int64_t, double, std::string, PList /*list*/, std::pair<int, PList> /*tuple*/, PDict, PTensor,
               PStorage, PGlobal>
      v;
};
using P = std::shared_ptr<PVal>;

template <typename T>
P mk(T&& v) {
  auto p = std::make_shared<PVal>();
  p->v = std::forward<T>(v);
  return p;
}
P tuple(PList items) { return mk(std::pair<int, PList>{0, std::move(items)}); }
const PList* asTuple(const P& p) {
  auto* t = std::get_if<std::pair<int, PList>>(&p->v);
  return t ? &t->second : nullptr;
}
const PList* asSeq(const P& p) {
  if (auto* t = asTuple(p)) return t;
  return std::get_if<PList>(&p->v);
}
std::vector<int64_t> ints(const P& p) {
  std::vector<int64_t> out;
  if (auto* s = asSeq(p)) {
    for (auto& x : *s) {
      if (auto* i = std::get_if<int64_t>(&x->v)) out.push_back(*i);
    }
  }
  return out;
}

std::string storageDtype(const std::string& name) {
  static const std::map<std::string, std::string> m{{"HalfStorage", "float16"},   {"FloatStorage", "float32"},
                                                    {"BFloat16Storage", "bfloat16"}, {"DoubleStorage", "float64"},
                                                    {"LongStorage", "int64"},     {"IntStorage", "int32"}};
  auto it = m.find(name);
  return it == m.end() ? "unknown" : it->second;
}

P unpickle(const std::vector<uint8_t>& d) {
  std::vector<P> stack;
  std::vector<size_t> marks;
  std::map<uint32_t, P> memo;
  size_t i = 0;
  auto need = [&](size_t n) {
    if (i + n > d.size()) throw std::runtime_error("truncated pickle");
  };
  auto pop = [&]() {
    if (stack.empty()) throw std::runtime_error("pickle stack underflow");
    P v = stack.back();
    stack.pop_back();
    return v;
  };
  auto popMark = [&]() {
    if (marks.empty()) throw std::runtime_error("pickle MARK missing");
    const size_t m = marks.back();
    marks.pop_back();
    PList items(stack.begin() + static_cast<std::ptrdiff_t>(m), stack.end());
    stack.resize(m);
    return items;
  };
  auto str = [&](size_t n) {
    need(n);
    std::string s(reinterpret_cast<const char*>(&d[i]), n);
    i += n;
    return s;
  };
  auto setItems = [&](const P& dict, const PList& kv) {
    auto* dd = std::get_if<PDict>(&dict->v);
    if (!dd) return;  // setting items on an opaque object: ignore
    for (size_t k = 0; k + 1 < kv.size(); k += 2) dd->items.emplace_back(kv[k], kv[k + 1]);
  };
  auto reduce = [&](const P& fn, const P& args) -> P {
    const auto* g = std::get_if<PGlobal>(&fn->v);
    const PList* a = asTuple(args);
    if (g && a) {
      if (g->module == "collections" && g->name == "OrderedDict") return mk(PDict{});
      if (g->module == "torch._utils" && (g->name == "_rebuild_tensor_v2" || g->name == "_rebuild_tensor") && a->size() >= 4) {
        PTensor t;
        if (auto* s = std::get_if<PStorage>(&(*a)[0]->v)) {
          t.storageKey = s->key;
          t.dtype = s->dtype;
        }
        if (auto* o = std::get_if<int64_t>(&(*a)[1]->v)) t.offset = *o;
        t.shape = ints((*a)[2]);
        t.stride = ints((*a)[3]);
        return mk(std::move(t));
      }
      if (g->module == "torch._utils" && g->name == "_rebuild_parameter" && !a->empty()) return (*a)[0];
    }
    return mk(PNone{});  // unknown callable: opaque value, never executed
  };

  while (true) {
    need(1);
    const uint8_t op = d[i++];
    switch (op) {
      case 0x80: need(1); ++i; break;                                  // PROTO
      case 0x95: need(8); i += 8; break;                               // FRAME
      case '.': return stack.empty() ? mk(PNone{}) : stack.back();     // STOP
      case '(': marks.push_back(stack.size()); break;                  // MARK
      case '}': stack.push_back(mk(PDict{})); break;                   // EMPTY_DICT
      case ']': stack.push_back(mk(PList{})); break;                   // EMPTY_LIST
      case ')': stack.push_back(tuple({})); break;                     // EMPTY_TUPLE
      case 'N': stack.push_back(mk(PNone{})); break;
      case 0x88: stack.push_back(mk(true)); break;                     // NEWTRUE
      case 0x89: stack.push_back(mk(false)); break;                    // NEWFALSE
      case 'K': need(1); stack.push_back(mk(static_cast<int64_t>(d[i]))); i += 1; break;                 // BININT1
      case 'M': need(2); stack.push_back(mk(static_cast<int64_t>(rd16(&d[i])))); i += 2; break;          // BININT2
      case 'J': need(4); stack.push_back(mk(static_cast<int64_t>(static_cast<int32_t>(rd32(&d[i]))))); i += 4; break;  // BININT
      case 0x8a: {                                                     // LONG1
        need(1);
        const size_t n = d[i++];
        need(n);
        int64_t v = 0;
        for (size_t k = 0; k < n && k < 8; ++k) v |= static_cast<int64_t>(d[i + k]) << (8 * k);
        if (n > 0 && n < 8 && (d[i + n - 1] & 0x80)) v -= static_cast<int64_t>(1) << (8 * n);
        i += n;
        stack.push_back(mk(v));
        break;
      }
      case 'G': {                                                      // BINFLOAT (big-endian)
        need(8);
        uint64_t b = 0;
        for (int k = 0; k < 8; ++k) b = (b << 8) | d[i + static_cast<size_t>(k)];
        double v;
        std::memcpy(&v, &b, 8);
        i += 8;
        stack.push_back(mk(v));
        break;
      }
      case 'X': { need(4); const uint32_t n = rd32(&d[i]); i += 4; stack.push_back(mk(str(n))); break; }  // BINUNICODE
      case 0x8c: { need(1); const size_t n = d[i++]; stack.push_back(mk(str(n))); break; }             // SHORT_BINUNICODE
      case 0x8d: { need(8); const uint64_t n = rd64(&d[i]); i += 8; stack.push_back(mk(str(n))); break; }  // BINUNICODE8
      case 'T': { need(4); const uint32_t n = rd32(&d[i]); i += 4; stack.push_back(mk(str(n))); break; }  // BINSTRING
      case 'U': { need(1); const size_t n = d[i++]; stack.push_back(mk(str(n))); break; }               // SHORT_BINSTRING
      case 'B': { need(4); const uint32_t n = rd32(&d[i]); i += 4; stack.push_back(mk(str(n))); break; }  // BINBYTES
      case 'C': { need(1); const size_t n = d[i++]; stack.push_back(mk(str(n))); break; }               // SHORT_BINBYTES
      case 't': stack.push_back(tuple(popMark())); break;              // TUPLE
      case 0x85: { auto a = pop(); stack.push_back(tuple({a})); break; }
      case 0x86: { auto b = pop(), a = pop(); stack.push_back(tuple({a, b})); break; }
      case 0x87: { auto c = pop(), b = pop(), a = pop(); stack.push_back(tuple({a, b, c})); break; }
      case 'l': stack.push_back(mk(popMark())); break;                 // LIST
      case 'd': { auto items = popMark(); auto dict = mk(PDict{}); setItems(dict, items); stack.push_back(dict); break; }  // DICT
      case 'a': { auto v = pop(); if (auto* l = std::get_if<PList>(&stack.back()->v)) l->push_back(v); break; }       // APPEND
      case 'e': { auto items = popMark(); if (auto* l = std::get_if<PList>(&stack.back()->v)) l->insert(l->end(), items.begin(), items.end()); break; }
      case 's': { auto v = pop(), k = pop(); setItems(stack.back(), {k, v}); break; }      // SETITEM
      case 'u': { auto items = popMark(); setItems(stack.back(), items); break; }          // SETITEMS
      case 'c': {                                                      // GLOBAL "module\nname\n"
        std::string mod, name;
        while (i < d.size() && d[i] != '\n') mod.push_back(static_cast<char>(d[i++]));
        ++i;
        while (i < d.size() && d[i] != '\n') name.push_back(static_cast<char>(d[i++]));
        ++i;
        stack.push_back(mk(PGlobal{mod, name}));
        break;
      }
      case 0x93: {                                                     // STACK_GLOBAL
        auto name = pop(), mod = pop();
        auto* m = std::get_if<std::string>(&mod->v);
        auto* n = std::get_if<std::string>(&name->v);
        stack.push_back(mk(PGlobal{m ? *m : "", n ? *n : ""}));
        break;
      }
      case 'R': { auto args = pop(), fn = pop(); stack.push_back(reduce(fn, args)); break; }   // REDUCE
      case 0x81: { auto args = pop(), cls = pop(); stack.push_back(reduce(cls, args)); break; }  // NEWOBJ
      case 'b': pop(); break;                                          // BUILD: state is not needed
      case 'Q': {                                                      // BINPERSID ('storage', type, key, location, numel)
        auto pid = pop();
        const PList* t = asTuple(pid);
        PStorage s;
        if (t && t->size() >= 3) {
          if (auto* g = std::get_if<PGlobal>(&(*t)[1]->v)) s.dtype = storageDtype(g->name);
          if (auto* k = std::get_if<std::string>(&(*t)[2]->v)) s.key = *k;
        }
        stack.push_back(mk(std::move(s)));
        break;
      }
      case 'q': need(1); memo[d[i]] = stack.back(); i += 1; break;     // BINPUT
      case 'r': need(4); memo[rd32(&d[i])] = stack.back(); i += 4; break;  // LONG_BINPUT
      case 0x94: memo[static_cast<uint32_t>(memo.size())] = stack.back(); break;  // MEMOIZE
      case 'h': need(1); stack.push_back(memo.at(d[i])); i += 1; break;    // BINGET
      case 'j': need(4); stack.push_back(memo.at(rd32(&d[i]))); i += 4; break;  // LONG_BINGET
      case '0': pop(); break;                                          // POP
      case '1': popMark(); break;                                      // POP_MARK
      case '2': stack.push_back(stack.back()); break;                  // DUP
      default:
        throw std::runtime_error("unsupported pickle opcode 0x" + std::to_string(op) + " (not a PyTorch checkpoint?)");
    }
  }
}

const P* dictGet(const P& dict, const std::string& key) {
  auto* d = std::get_if<PDict>(&dict->v);
  if (!d) return nullptr;
  for (auto& [k, v] : d->items) {
    if (auto* s = std::get_if<std::string>(&k->v); s && *s == key) return &v;
  }
  return nullptr;
}

// ------------------------------------------------------------ checkpoint

struct Checkpoint {
  std::unique_ptr<ZipReader> zip;   // the .pth itself (a ZIP)
  std::string prefix;               // "archive/" etc.
  P root;
  RvcCheckpointInfo info;
  std::map<std::string, PTensor> tensors;
  std::string tempPth;              // extracted from an outer .zip
  ~Checkpoint() {
    if (!tempPth.empty()) std::remove(tempPth.c_str());
  }
};

std::unique_ptr<Checkpoint> openCheckpoint(const std::string& path, const std::string& scratchDir) {
  auto ck = std::make_unique<Checkpoint>();
  auto outer = std::make_unique<ZipReader>(path);
  const bool isTorch = std::any_of(outer->entries().begin(), outer->entries().end(),
                                   [](const ZipReader::Entry& e) { return endsWith(e.name, "data.pkl"); });
  if (isTorch) {
    ck->zip = std::move(outer);
    ck->info.pthName = path.substr(path.find_last_of('/') + 1);
  } else {
    const ZipReader::Entry* pth = nullptr;
    for (const auto& e : outer->entries()) {
      if (endsWith(e.name, ".pth") && e.name.find("__MACOSX") == std::string::npos && (!pth || e.size > pth->size)) pth = &e;
      if (endsWith(e.name, ".index")) ck->info.hasIndex = true;
    }
    if (!pth) {
      for (const auto& e : outer->entries()) {
        if (endsWith(e.name, ".onnx")) throw std::runtime_error("this archive contains an ONNX voice; import the .onnx directly");
      }
      throw std::runtime_error("no .pth voice found in the archive");
    }
    ck->info.pthName = pth->name.substr(pth->name.find_last_of('/') + 1);
    ck->tempPth = (scratchDir.empty() ? std::string("/tmp") : scratchDir) + "/mobigpt-import-" + std::to_string(pth->localHeaderOffset) + ".pth";
    outer->extract(*pth, ck->tempPth);
    ck->zip = std::make_unique<ZipReader>(ck->tempPth);
  }
  const ZipReader::Entry* pkl = nullptr;
  for (const auto& e : ck->zip->entries()) {
    if (endsWith(e.name, "data.pkl")) pkl = &e;
  }
  if (!pkl) throw std::runtime_error("not a PyTorch checkpoint (no data.pkl)");
  ck->prefix = pkl->name.substr(0, pkl->name.size() - std::strlen("data.pkl"));
  ck->root = unpickle(ck->zip->read(*pkl));

  const P* weight = dictGet(ck->root, "weight");
  if (!weight) weight = dictGet(ck->root, "model");
  if (!weight) {
    if (dictGet(ck->root, "optimizer")) throw std::runtime_error("this is a training checkpoint (G_*.pth/D_*.pth); export the small inference model in RVC first");
    throw std::runtime_error("not an RVC voice: no 'weight' table");
  }
  for (auto& [k, v] : std::get<PDict>((*weight)->v).items) {
    auto* key = std::get_if<std::string>(&k->v);
    auto* t = std::get_if<PTensor>(&v->v);
    if (key && t) ck->tensors[*key] = *t;
  }
  auto& in = ck->info;
  in.tensors = ck->tensors.size();
  if (auto* v = dictGet(ck->root, "version")) {
    if (auto* s = std::get_if<std::string>(&(*v)->v)) in.version = *s;
  }
  if (auto* v = dictGet(ck->root, "f0")) {
    if (auto* i = std::get_if<int64_t>(&(*v)->v)) in.f0 = *i != 0;
  }
  if (auto* v = dictGet(ck->root, "info")) {
    if (auto* s = std::get_if<std::string>(&(*v)->v)) in.info = *s;
  }
  if (auto* v = dictGet(ck->root, "config")) {
    if (auto* seq = asSeq(*v); seq && !seq->empty()) {
      if (auto* sr = std::get_if<int64_t>(&seq->back()->v)) in.sampleRate = static_cast<int>(*sr);
    }
  }
  if (in.sampleRate == 0) {
    if (auto* v = dictGet(ck->root, "sr")) {
      if (auto* s = std::get_if<std::string>(&(*v)->v)) in.sampleRate = std::atoi(s->c_str()) * 1000;
    }
  }
  if (auto it = ck->tensors.find("enc_p.emb_phone.weight"); it != ck->tensors.end() && it->second.shape.size() == 2) {
    in.featureDim = static_cast<int>(it->second.shape[1]);
  }
  if (in.version.empty()) in.version = in.featureDim == 768 ? "v2" : "v1";
  if (auto it = ck->tensors.find("emb_g.weight"); it != ck->tensors.end() && !it->second.shape.empty()) {
    in.speakers = static_cast<int>(it->second.shape[0]);
  }
  if (!ck->tensors.empty()) in.dtype = ck->tensors.begin()->second.dtype;
  return ck;
}

uint16_t bf16ToHalf(uint16_t b) {
  uint32_t bits = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return floatToHalf(f);
}

// ONNX ModelProto.metadata_props (field 14) by a minimal protobuf scan.
std::map<std::string, std::string> onnxMetadata(const std::string& path) {
  const auto size = fileSize(path);
  auto d = readRange(path, 0, size);
  auto varint = [&](size_t& i) {
    uint64_t v = 0;
    for (int s = 0; i < d.size(); s += 7) {
      const uint8_t b = d[i++];
      v |= static_cast<uint64_t>(b & 0x7f) << s;
      if (!(b & 0x80)) break;
    }
    return v;
  };
  std::map<std::string, std::string> meta;
  size_t i = 0;
  while (i < d.size()) {
    const uint64_t tag = varint(i);
    const uint32_t field = static_cast<uint32_t>(tag >> 3), wire = tag & 7;
    if (wire == 0) {
      varint(i);
    } else if (wire == 1) {
      i += 8;
    } else if (wire == 5) {
      i += 4;
    } else if (wire == 2) {
      const uint64_t len = varint(i);
      if (field == 14) {
        size_t j = i, end = i + len;
        std::string key, value;
        while (j < end) {
          const uint64_t t = varint(j);
          const uint64_t l = varint(j);
          std::string s(reinterpret_cast<const char*>(&d[j]), l);
          j += l;
          if ((t >> 3) == 1) key = s;
          if ((t >> 3) == 2) value = s;
        }
        meta[key] = value;
      }
      i += len;
    } else {
      throw std::runtime_error("unexpected protobuf wire type in " + path);
    }
  }
  return meta;
}

}  // namespace

RvcCheckpointInfo inspectRvcCheckpoint(const std::string& pthOrZip) {
  const std::string dir = pthOrZip.substr(0, pthOrZip.find_last_of('/'));
  return openCheckpoint(pthOrZip, dir)->info;
}

std::string rvcTemplateName(const RvcCheckpointInfo& info) {
  return "rvc_template_" + info.version + "_" + std::to_string(info.sampleRate / 1000) + "k.onnx";
}

RvcImportResult importRvcCheckpoint(const std::string& pthOrZip, const std::string& templatePath, const std::string& outDir) {
  auto ck = openCheckpoint(pthOrZip, outDir);
  if (!ck->info.f0) throw std::runtime_error("voices without pitch guidance (f0 = 0) are not supported yet");
  const auto meta = onnxMetadata(templatePath);
  auto get = [&](const char* k) {
    auto it = meta.find(k);
    if (it == meta.end()) throw std::runtime_error(std::string("not a MobiGPT voice template (missing ") + k + ")");
    return it->second;
  };
  if (get("version") != ck->info.version || std::stoi(get("sample_rate")) != ck->info.sampleRate) {
    throw std::runtime_error("template " + templatePath + " does not match a " + ck->info.version + " " +
                             std::to_string(ck->info.sampleRate) + " Hz voice");
  }
  const uint64_t total = std::stoull(get("mobigpt_weights_bytes"));
  const int64_t speakerRows = std::stoll(get("mobigpt_speaker_rows"));
  std::vector<uint8_t> out(total, 0);

  std::string manifest = get("mobigpt_manifest");
  size_t pos = 0, written = 0;
  while (pos < manifest.size()) {
    size_t eol = manifest.find('\n', pos);
    if (eol == std::string::npos) eol = manifest.size();
    const std::string line = manifest.substr(pos, eol - pos);
    pos = eol + 1;
    const size_t t1 = line.find('\t'), t2 = line.find('\t', t1 + 1);
    if (t1 == std::string::npos || t2 == std::string::npos) continue;
    const std::string key = line.substr(0, t1);
    const uint64_t offset = std::stoull(line.substr(t1 + 1, t2 - t1 - 1));
    std::vector<int64_t> dims;
    for (size_t s = t2 + 1; s < line.size();) {
      size_t c = line.find(',', s);
      if (c == std::string::npos) c = line.size();
      dims.push_back(std::stoll(line.substr(s, c - s)));
      s = c + 1;
    }
    auto it = ck->tensors.find(key);
    if (it == ck->tensors.end()) throw std::runtime_error("voice is missing tensor " + key + " (unsupported RVC variant)");
    const PTensor& t = it->second;
    // Shapes must match, except the speaker table, which is padded or truncated.
    const bool speakerTable = key == "emb_g.weight";
    if (t.shape.size() != dims.size()) throw std::runtime_error("unexpected rank for " + key);
    for (size_t k = speakerTable ? 1 : 0; k < dims.size(); ++k) {
      if (t.shape[k] != dims[k]) throw std::runtime_error("unexpected shape for " + key + " (unsupported RVC variant)");
    }
    // Contiguous row-major only (what torch.save writes for parameters).
    int64_t expect = 1;
    for (size_t k = t.shape.size(); k-- > 0;) {
      if (t.shape[k] > 1 && t.stride.size() == t.shape.size() && t.stride[k] != expect) throw std::runtime_error("non-contiguous tensor " + key);
      expect *= t.shape[k];
    }
    const int64_t rows = speakerTable ? std::min<int64_t>(t.shape[0], speakerRows) : (t.shape.empty() ? 1 : t.shape[0]);
    int64_t inner = 1;
    for (size_t k = 1; k < t.shape.size(); ++k) inner *= t.shape[k];
    const int64_t copyElems = t.shape.empty() ? 1 : rows * inner;
    const size_t elem = t.dtype == "float32" ? 4 : 2;
    if (t.dtype != "float16" && t.dtype != "float32" && t.dtype != "bfloat16") throw std::runtime_error("unsupported tensor type " + t.dtype + " for " + key);

    const auto* entry = ck->zip->find(ck->prefix + "data/" + t.storageKey);
    if (!entry) throw std::runtime_error("missing storage " + t.storageKey + " for " + key);
    // Storages are written stored (uncompressed) by torch.save; read just this tensor's slice.
    std::vector<uint8_t> raw;
    if (entry->method == 0) {
      auto h = readRange(ck->zip->path(), entry->localHeaderOffset, 30);
      const uint64_t base = entry->localHeaderOffset + 30 + rd16(&h[26]) + rd16(&h[28]);
      raw = readRange(ck->zip->path(), base + static_cast<uint64_t>(t.offset) * elem, static_cast<uint64_t>(copyElems) * elem);
    } else {
      auto all = ck->zip->read(*entry);
      const size_t from = static_cast<size_t>(t.offset) * elem, n = static_cast<size_t>(copyElems) * elem;
      if (from + n > all.size()) throw std::runtime_error("storage too small for " + key);
      raw.assign(all.begin() + static_cast<std::ptrdiff_t>(from), all.begin() + static_cast<std::ptrdiff_t>(from + n));
    }
    if (offset + static_cast<uint64_t>(copyElems) * 2 > total) throw std::runtime_error("template layout overflow at " + key);
    uint8_t* dst = out.data() + offset;
    if (t.dtype == "float16") {
      std::memcpy(dst, raw.data(), raw.size());
    } else {
      for (int64_t k = 0; k < copyElems; ++k) {
        uint16_t h;
        if (t.dtype == "float32") {
          float f;
          std::memcpy(&f, &raw[static_cast<size_t>(k) * 4], 4);
          h = floatToHalf(f);
        } else {
          uint16_t b;
          std::memcpy(&b, &raw[static_cast<size_t>(k) * 2], 2);
          h = bf16ToHalf(b);
        }
        std::memcpy(dst + k * 2, &h, 2);
      }
    }
    ++written;
  }
  if (written == 0) throw std::runtime_error("template manifest is empty");

  const std::string weightsPath = outDir + "/weights.bin";
  {
    std::ofstream w(weightsPath, std::ios::binary | std::ios::trunc);
    if (!w) throw std::runtime_error("cannot write " + weightsPath);
    w.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
    if (!w) throw std::runtime_error("disk full while writing " + weightsPath);
  }
  const std::string modelPath = outDir + "/model.onnx";
  {
    std::ifstream src(templatePath, std::ios::binary);
    std::ofstream dst(modelPath, std::ios::binary | std::ios::trunc);
    dst << src.rdbuf();
    if (!dst) throw std::runtime_error("cannot write " + modelPath);
  }
  RvcImportResult r;
  r.modelPath = modelPath;
  r.weightsBytes = total;
  r.info = ck->info;
  return r;
}

}  // namespace mobigpt::rvc
