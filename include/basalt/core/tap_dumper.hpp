// include/basalt/core/tap_dumper.hpp - append-only writer for the DFlash2 retrain's tap data.
//
// Writes a raw binary file (the `.taps` format) that the training script reads with a small
// NumPy loader.  The format is a header followed by a sequence of records, each holding a
// chunk's token ids + taps.  A pure write; the engine's arithmetic is untouched.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace basalt::core {

/// Append-only writer for the `--dump-taps` flag.  Writes a raw binary file:
///
///   Header (20 bytes):
///     magic    "TAPS" (4 bytes)
///     version  u32 (= 1)
///     n_taps   u32 (the number of tapped layers, e.g. 5)
///     n_embd   u32 (the embedding width, e.g. 2560)
///     dtype    u32 (1 = BF16)
///
///   Records (repeated):
///     n_tokens  u32
///     tokens    n_tokens × i32
///     taps      n_tokens × n_taps × n_embd × u16 (BF16)
///
/// The training script concatenates the records into `tokens[N]` + `taps[N, n_taps, n_embd]`.
class TapDumper {
public:
    /// Opens the file for writing.  `n_taps` and `n_embd` are written to the header.
    /// Returns false with `err` set when the file cannot be opened.
    bool open(const std::string& path, int n_taps, int n_embd, std::string& err) {
        n_taps_ = n_taps;
        n_embd_ = n_embd;
        f_ = std::fopen(path.c_str(), "wb");
        if (!f_) {
            err = "tap dumper: cannot open " + path;
            return false;
        }
        // Write the header
        const char magic[4] = {'T', 'A', 'P', 'S'};
        const uint32_t version = 1;
        const uint32_t dtype = 1;  // BF16
        std::fwrite(magic, 1, 4, f_);
        std::fwrite(&version, 4, 1, f_);
        std::fwrite(&n_taps_, 4, 1, f_);
        std::fwrite(&n_embd_, 4, 1, f_);
        std::fwrite(&dtype, 4, 1, f_);
        return true;
    }

    /// Writes one record: `n_tokens` token ids + their taps (BF16, [n_tokens, n_taps, n_embd]).
    /// The taps are read from a device or host buffer; this is a pure write.
    void write_record(const int32_t* tokens, const uint16_t* taps_bf16, int n_tokens) {
        if (!f_) return;
        tokens_total_ += n_tokens;
        const uint32_t nt = (uint32_t) n_tokens;
        std::fwrite(&nt, 4, 1, f_);
        std::fwrite(tokens, 4, nt, f_);
        const size_t n_tap_bytes = (size_t) n_tokens * n_taps_ * n_embd_ * 2;
        std::fwrite(taps_bf16, 1, n_tap_bytes, f_);
    }

    /// Writes one record from an F32 taps buffer (converts F32 → BF16 inline).
    void write_record_f32(const int32_t* tokens, const float* taps_f32, int n_tokens) {
        if (!f_) return;
        tokens_total_ += n_tokens;
        const uint32_t nt = (uint32_t) n_tokens;
        std::fwrite(&nt, 4, 1, f_);
        std::fwrite(tokens, 4, nt, f_);
        // Convert F32 → BF16 and write
        const size_t n_elems = (size_t) n_tokens * n_taps_ * n_embd_;
        // Process in chunks to avoid a huge stack allocation
        constexpr size_t kChunk = 65536;
        size_t done = 0;
        while (done < n_elems) {
            const size_t k = std::min(kChunk, n_elems - done);
            uint16_t buf[kChunk];
            for (size_t i = 0; i < k; ++i) {
                // F32 → BF16: round to nearest even (upper 16 bits)
                uint32_t u;
                std::memcpy(&u, &taps_f32[done + i], 4);
                u = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
                buf[i] = (uint16_t) u;
            }
            std::fwrite(buf, 2, k, f_);
            done += k;
        }
    }

    /// Closes the file.  Safe to call multiple times.
    void close() {
        if (f_) { std::fclose(f_); f_ = nullptr; }
    }

    bool open() const { return f_ != nullptr; }
    int64_t tokens_written() const { return tokens_total_; }

    ~TapDumper() { close(); }
    TapDumper() = default;
    TapDumper(const TapDumper&) = delete;
    TapDumper& operator=(const TapDumper&) = delete;

private:
    FILE* f_ = nullptr;
    int n_taps_ = 0, n_embd_ = 0;
    int64_t tokens_total_ = 0;
};

}  // namespace basalt::core
