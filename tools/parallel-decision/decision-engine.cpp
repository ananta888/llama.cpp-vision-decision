#include "decision-engine.h"

#include "chat.h"
#include "common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>

namespace llama_decision {

namespace {

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

struct decision_field {
    tokens_t              suffix;
    std::vector<tokens_t> paths;

    std::vector<int> active;  // greedy walk
    tokens_t         chosen;

    std::vector<tokens_t> node_prefix;   // tree: trie nodes with more than one allowed next token
    std::vector<tokens_t> node_options;
    std::vector<float>    probs;

    bool  use_tree     = false;
    float temperature  = 1.0f;
    int   winner       = -1;
    float path_score   = 1.0f;
    int   scored_nodes = 0;

    decision_field(tokens_t s, std::vector<tokens_t> p) : suffix(std::move(s)), paths(std::move(p)) {
        for (int i = 0; i < (int) paths.size(); ++i) {
            active.push_back(i);
        }
    }

    // Greedy: follow single-path edges for free; return the options at the next divergence.
    tokens_t options() {
        while (active.size() > 1) {
            const size_t depth = chosen.size();
            tokens_t opts;
            for (int i : active) {
                if (depth >= paths[i].size()) {
                    throw std::runtime_error("candidate paths must be distinct and terminated");
                }
                if (std::find(opts.begin(), opts.end(), paths[i][depth]) == opts.end()) {
                    opts.push_back(paths[i][depth]);
                }
            }
            if (opts.size() > 1) {
                return opts;
            }
            chosen.push_back(opts[0]);
        }
        winner = active.empty() ? -1 : active[0];
        return {};
    }

    void select(llama_token tok, float p) {
        const size_t depth = chosen.size();
        std::vector<int> remaining;
        for (int i : active) {
            if (paths[i][depth] == tok) {
                remaining.push_back(i);
            }
        }
        active = remaining;
        chosen.push_back(tok);
        path_score *= p;
        scored_nodes += 1;
    }

    void build_nodes() {
        std::vector<std::pair<std::vector<int>, size_t>> stack;
        std::vector<int> root(paths.size());
        for (int i = 0; i < (int) paths.size(); ++i) {
            root[i] = i;
        }
        stack.push_back({ root, 0 });
        while (!stack.empty()) {
            auto [act, depth] = stack.back();
            stack.pop_back();
            if (act.size() <= 1) {
                continue;
            }
            tokens_t opts;
            for (int i : act) {
                if (depth >= paths[i].size()) {
                    throw std::runtime_error("candidate paths must be distinct and terminated");
                }
                if (std::find(opts.begin(), opts.end(), paths[i][depth]) == opts.end()) {
                    opts.push_back(paths[i][depth]);
                }
            }
            if (opts.size() > 1) {
                node_prefix.emplace_back(paths[act[0]].begin(), paths[act[0]].begin() + depth);
                node_options.push_back(opts);
            }
            for (llama_token tok : opts) {
                std::vector<int> sub;
                for (int i : act) {
                    if (paths[i][depth] == tok) {
                        sub.push_back(i);
                    }
                }
                stack.push_back({ sub, depth + 1 });
            }
        }
    }

    int tree_rows() const {
        int n = 0;
        for (const auto & p : node_prefix) {
            n += (int) (suffix.size() + p.size());
        }
        return n;
    }

    // Exact constrained distribution: log-softmax at each node over its allowed tokens,
    // summed along every candidate path, normalised over candidates.
    void finish_tree(const std::vector<std::vector<float>> & node_scores) {
        std::vector<std::vector<float>> node_logp;
        for (const auto & s : node_scores) {
            const float mx = *std::max_element(s.begin(), s.end());
            double z = 0;
            for (float x : s) {
                z += std::exp(x - mx);
            }
            const float lz = mx + (float) std::log(z);
            std::vector<float> lp;
            for (float x : s) {
                lp.push_back(x - lz);
            }
            node_logp.push_back(lp);
        }
        std::vector<float> path_lp(paths.size(), 0.0f);
        for (size_t i = 0; i < paths.size(); ++i) {
            for (size_t n = 0; n < node_prefix.size(); ++n) {
                const auto & pre = node_prefix[n];
                if (pre.size() >= paths[i].size() || !std::equal(pre.begin(), pre.end(), paths[i].begin())) {
                    continue;
                }
                const auto & opts = node_options[n];
                const auto   it   = std::find(opts.begin(), opts.end(), paths[i][pre.size()]);
                if (it == opts.end()) {
                    throw std::runtime_error("candidate token missing from its trie node");
                }
                path_lp[i] += node_logp[n][it - opts.begin()];
            }
        }
        for (float & x : path_lp) {
            x /= temperature;
        }
        const int best = (int) (std::max_element(path_lp.begin(), path_lp.end()) - path_lp.begin());
        double z = 0;
        for (float x : path_lp) {
            z += std::exp(x - path_lp[best]);
        }
        probs.clear();
        for (float x : path_lp) {
            probs.push_back((float) (std::exp(x - path_lp[best]) / z));
        }
        winner       = best;
        path_score   = probs[best];
        scored_nodes = (int) node_prefix.size();
    }
};

} // namespace

// ---------------------------------------------------------------- engine

engine::engine(llama_context * ctx, llama_seq_id seq_base, int n_seqs)
    : ctx(ctx), vocab(llama_model_get_vocab(llama_get_model(ctx))), mem(llama_get_memory(ctx)),
      seq_snap(seq_base), seq_pool(seq_base + 1), n_pool(n_seqs - 1) {
    if (n_seqs < 3) {
        throw std::invalid_argument("a decision engine needs at least 3 sequences");
    }
}

tokens_t engine::tokenize(const std::string & text, bool add_special) const {
    tokens_t toks = common_tokenize(vocab, text, add_special, /*parse_special=*/ true);
    // a chat template may already start with the BOS text; keep a single BOS
    const llama_token bos = llama_vocab_bos(vocab);
    if (toks.size() >= 2 && toks[0] == bos && toks[1] == bos) {
        toks.erase(toks.begin());
    }
    return toks;
}

// Decode several prompts, each on its own sequence, packed into as few batches as n_batch allows.
void engine::decode_parts(const std::vector<prompt_part> & parts) {
    const int n_batch = (int) llama_n_batch(ctx);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    auto flush = [&]() {
        const int rc = batch.n_tokens > 0 ? llama_decode(ctx, batch) : 0;
        common_batch_clear(batch);
        if (rc != 0) {
            llama_batch_free(batch);
            throw std::runtime_error(rc == 1 ? "no free KV cache space for the decision prompt"
                                             : "llama_decode failed on the decision prompt (" + std::to_string(rc) + ")");
        }
    };
    for (const auto & p : parts) {
        for (size_t i = 0; i < p.toks->size(); ++i) {
            if (batch.n_tokens == n_batch) {
                flush();
            }
            common_batch_add(batch, (*p.toks)[i], p.pos0 + (llama_pos) i, { p.seq }, false);
        }
    }
    flush();
    llama_batch_free(batch);
}

// Restore (or build) the cached static prefix on seq_snap. Only this engine's own sequences are
// touched, so it can share a context with other users (e.g. server slots).
bool engine::prepare_prefix(const tokens_t & shared, bool allow_cache) {
    if (allow_cache && !shared.empty() && shared == cached &&
        llama_memory_seq_pos_max(mem, seq_snap) == (llama_pos) cached.size() - 1) {
        return true;
    }
    for (llama_seq_id s = seq_snap; s < seq_pool + n_pool; ++s) {
        llama_memory_seq_rm(mem, s, -1, -1);
    }
    cached.clear();
    if (!shared.empty()) {
        decode_parts({ { &shared, 0, seq_snap } });
        cached = shared;
    }
    return false;
}

// Free the trunk and branch sequences; the cached prefix on seq_snap stays.
void engine::clear_pool() {
    for (llama_seq_id s = seq_pool; s < seq_pool + n_pool; ++s) {
        llama_memory_seq_rm(mem, s, -1, -1);
    }
}

// Score each branch as its own sequence forked from its trunk; return each branch's last-token
// logits restricted to its candidate tokens. Groups are bounded by free sequences and batch rows.
std::vector<std::vector<float>> engine::score_branches(const std::vector<branch> & branches, llama_seq_id first, int n_free) {
    std::vector<std::vector<float>> result(branches.size());
    const int max_rows = (int) llama_n_batch(ctx);
    size_t start = 0;
    while (start < branches.size()) {
        size_t end  = start;
        int    rows = 0;
        while (end < branches.size() && (int) (end - start) < n_free && rows + (int) branches[end].toks.size() <= max_rows) {
            rows += (int) branches[end].toks.size();
            ++end;
        }
        if (end == start) {
            throw std::runtime_error("a decision suffix exceeds the batch size");
        }
        llama_batch batch = llama_batch_init(rows, 0, 1);
        std::vector<int> out_idx;
        for (size_t b = start; b < end; ++b) {
            const llama_seq_id seq = first + (llama_seq_id) (b - start);
            llama_memory_seq_rm(mem, seq, -1, -1);
            llama_memory_seq_cp(mem, branches[b].trunk, seq, -1, -1);
            const auto & toks = branches[b].toks;
            for (size_t i = 0; i < toks.size(); ++i) {
                const bool last = i + 1 == toks.size();
                if (last) {
                    out_idx.push_back(batch.n_tokens);
                }
                common_batch_add(batch, toks[i], branches[b].pos0 + (llama_pos) i, { seq }, last);
            }
        }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            throw std::runtime_error(rc == 1 ? "no free KV cache space for the decision branches"
                                             : "llama_decode failed on the decision branches (" + std::to_string(rc) + ")");
        }
        for (size_t b = start; b < end; ++b) {
            const float * logits = llama_get_logits_ith(ctx, out_idx[b - start]);
            for (llama_token t : branches[b].cands) {
                result[b].push_back(logits[t]);
            }
            llama_memory_seq_rm(mem, first + (llama_seq_id) (b - start), -1, -1);
        }
        start = end;
    }
    return result;
}

result engine::decide(const std::string & shared_text, const std::string & context_text,
                      const std::vector<field_input> & inputs, const options & opt) {
    batch_result b = decide_batch(shared_text, { context_text }, inputs, opt);
    result r = std::move(b.items[0]);
    r.cache_hit     = b.cache_hit;
    r.shared_tokens = b.shared_tokens;
    r.rounds        = b.rounds;
    r.prefill_ms    = b.prefill_ms;
    r.scoring_ms    = b.scoring_ms;
    return r;
}

batch_result engine::decide_batch(const std::string & shared_text, const std::vector<std::string> & contexts,
                                  const std::vector<field_input> & inputs, const options & opt) {
    std::vector<context_input> in(contexts.size());
    for (size_t i = 0; i < contexts.size(); ++i) {
        in[i].text = contexts[i];
    }
    return decide_batch(shared_text, in, inputs, opt);
}

batch_result engine::decide_batch(const std::string & shared_text, const std::vector<context_input> & contexts,
                                  const std::vector<field_input> & inputs, const options & opt) {
    if (opt.mode != "auto" && opt.mode != "tree" && opt.mode != "greedy") {
        throw std::invalid_argument("mode must be auto, tree or greedy");
    }
    if (contexts.empty()) {
        throw std::invalid_argument("a decision needs at least one context");
    }
    const tokens_t shared = tokenize(shared_text, true);
    std::vector<tokens_t> prefixes(contexts.size());
    for (size_t i = 0; i < contexts.size(); ++i) {
        if (contexts[i].prefill) {
            if (contexts[i].n_tokens == 0) {
                throw std::invalid_argument("the decision context must not be empty");
            }
            continue;
        }
        prefixes[i] = tokenize(contexts[i].text, shared.empty());
        if (prefixes[i].empty()) {
            throw std::invalid_argument("the decision context must not be empty");
        }
    }

    std::vector<decision_field> fields;
    int total    = 0;
    int branches = 0; // round-1 branches of one context
    for (const auto & in : inputs) {
        const size_t n = in.candidates.size();
        if (n < 1 || n > 255) {
            throw std::invalid_argument("each field needs 1-255 allowed values");
        }
        if (!(in.temperature > 0.0f)) {
            throw std::invalid_argument("a field temperature must be positive");
        }
        tokens_t suffix;
        std::vector<tokens_t> paths;
        if (opt.split_boundary) {
            for (const auto & c : in.candidates) {
                paths.push_back(tokenize(c + "\n", false));
            }
            suffix = tokenize(in.suffix, false);
        } else {
            // Tokenise each complete "suffix + value + terminator" and split at the longest token
            // prefix shared by every candidate: the value's first token is exactly what the model
            // would write itself (e.g. `":` then ` true`, not `": ` then `true`).
            std::vector<tokens_t> seqs;
            for (const auto & c : in.candidates) {
                seqs.push_back(tokenize(in.suffix + c + "\n", false));
            }
            size_t common = seqs[0].size();
            for (const auto & s : seqs) {
                size_t c = 0;
                while (c < std::min(common, s.size()) && s[c] == seqs[0][c]) {
                    ++c;
                }
                common = std::min({ common, c, s.size() - 1 });
            }
            suffix.assign(seqs[0].begin(), seqs[0].begin() + common);
            for (const auto & s : seqs) {
                paths.emplace_back(s.begin() + common, s.end());
            }
        }
        if (suffix.empty()) {
            throw std::invalid_argument("a field suffix must not be empty");
        }
        size_t max_path = 0;
        for (size_t a = 0; a < paths.size(); ++a) {
            max_path = std::max(max_path, paths[a].size());
            for (size_t b = 0; b < a; ++b) {
                const size_t m = std::min(paths[a].size(), paths[b].size());
                if (std::equal(paths[a].begin(), paths[a].begin() + m, paths[b].begin())) {
                    throw std::invalid_argument("two allowed values tokenise to colliding paths");
                }
            }
        }
        decision_field field(suffix, paths);
        field.temperature = in.temperature;
        field.build_nodes();
        field.use_tree = opt.mode == "tree" ? true : opt.mode == "greedy" ? false : n <= opt.tree_max;
        total    += field.use_tree ? field.tree_rows() : (int) (suffix.size() + max_path);
        branches += field.use_tree ? (int) field.node_prefix.size() : 1;
        fields.push_back(std::move(field));
    }

    batch_result out;
    out.shared_tokens = shared.size();
    out.rows          = total * (int) contexts.size();
    out.items.resize(contexts.size());

    // a group of trunks shares the KV cache with the cached prefix and one round of branch rows
    const size_t n_kv    = llama_n_ctx(ctx);
    const size_t n_batch = llama_n_batch(ctx);
    std::vector<size_t> ctx_tokens(contexts.size());
    for (size_t i = 0; i < contexts.size(); ++i) {
        ctx_tokens[i] = contexts[i].prefill ? contexts[i].n_tokens : prefixes[i].size();
        if (shared.size() + ctx_tokens[i] + std::min<size_t>(n_batch, total) > n_kv) {
            throw std::invalid_argument("a decision context of " + std::to_string(ctx_tokens[i]) + " tokens does not fit the KV cache (" +
                                        std::to_string(n_kv) + " cells, " + std::to_string(shared.size()) + " used by the instructions)");
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    out.cache_hit = prepare_prefix(shared, opt.allow_cache);
    out.prefill_ms += ms_since(t0);

    // on error, free the trunks and branches so they do not keep KV cells
    struct pool_guard {
        engine * eng;
        bool     armed = true;
        ~pool_guard() {
            if (armed) {
                eng->clear_pool();
            }
        }
    } guard { this };

    std::vector<llama_pos> pos_next(contexts.size());

    // each context in a group holds one trunk sequence; the rest of the pool scores branches
    const size_t per_group = std::clamp<size_t>(n_pool / (1 + branches), 1, contexts.size());
    // at most one prefilled (media) context per group, decoded first right after the prefix: media decoded
    // after other trunks' cells give slightly different numbers, text does not
    auto group_size = [&](size_t g0) {
        size_t n = 1, used = shared.size() + ctx_tokens[g0];
        int n_prefill = contexts[g0].prefill ? 1 : 0;
        while (g0 + n < contexts.size() && n < per_group && n_prefill + (contexts[g0 + n].prefill ? 1 : 0) <= 1 &&
               used + ctx_tokens[g0 + n] + std::min<size_t>(n_batch, (size_t) total * (n + 1)) <= n_kv) {
            used += ctx_tokens[g0 + n];
            n_prefill += contexts[g0 + n].prefill ? 1 : 0;
            ++n;
        }
        return n;
    };
    int n_groups = 0;
    for (size_t g0 = 0, n_group = 0; g0 < contexts.size(); g0 += n_group, ++n_groups) {
        n_group = group_size(g0);

        const auto tp = std::chrono::steady_clock::now();
        const llama_pos pos_ctx = (llama_pos) shared.size();
        std::vector<prompt_part> parts;
        for (size_t i = 0; i < n_group; ++i) {
            const llama_seq_id trunk = seq_pool + (llama_seq_id) i;
            llama_memory_seq_rm(mem, trunk, -1, -1);
            if (!shared.empty()) {
                llama_memory_seq_cp(mem, seq_snap, trunk, -1, -1);
            }
            if (!contexts[g0 + i].prefill) {
                parts.push_back({ &prefixes[g0 + i], pos_ctx, trunk });
                pos_next[g0 + i] = pos_ctx + (llama_pos) prefixes[g0 + i].size();
            }
        }
        for (size_t i = 0; i < n_group; ++i) {
            if (contexts[g0 + i].prefill) {
                pos_next[g0 + i] = contexts[g0 + i].prefill(seq_pool + (llama_seq_id) i, pos_ctx);
                if (pos_next[g0 + i] <= pos_ctx) {
                    throw std::runtime_error("the context prefill did not advance the position");
                }
            }
        }
        decode_parts(parts);
        llama_synchronize(ctx); // llama_decode is asynchronous: wait for the prefill so its time isn't billed to scoring
        out.prefill_ms += ms_since(tp);

        // round 1 carries every tree node and each greedy field's first step; later rounds only
        // continue greedy fields that are still open
        const auto ts = std::chrono::steady_clock::now();
        std::vector<std::vector<decision_field>> state(n_group, fields);
        bool first = true;
        while (true) {
            std::vector<branch> todo;
            std::vector<std::pair<size_t, size_t>> owner; // (context in group, field)
            for (size_t i = 0; i < n_group; ++i) {
                const llama_seq_id trunk = seq_pool + (llama_seq_id) i;
                const llama_pos    pos0  = pos_next[g0 + i];
                for (size_t f = 0; f < state[i].size(); ++f) {
                    auto & fd = state[i][f];
                    if (fd.use_tree) {
                        if (first) {
                            for (size_t n = 0; n < fd.node_prefix.size(); ++n) {
                                tokens_t ids = fd.suffix;
                                ids.insert(ids.end(), fd.node_prefix[n].begin(), fd.node_prefix[n].end());
                                todo.push_back({ trunk, pos0, ids, fd.node_options[n] });
                                owner.push_back({ i, f });
                            }
                        }
                    } else {
                        tokens_t opts = fd.options();
                        if (!opts.empty()) {
                            tokens_t ids = fd.suffix;
                            ids.insert(ids.end(), fd.chosen.begin(), fd.chosen.end());
                            todo.push_back({ trunk, pos0, ids, opts });
                            owner.push_back({ i, f });
                        }
                    }
                }
            }
            if (todo.empty()) {
                break;
            }
            const auto scores = score_branches(todo, seq_pool + (llama_seq_id) n_group, n_pool - (int) n_group);
            out.rounds += 1;
            std::vector<std::vector<std::vector<std::vector<float>>>> tree_scores(n_group, std::vector<std::vector<std::vector<float>>>(fields.size()));
            for (size_t row = 0; row < owner.size(); ++row) {
                const auto [i, f] = owner[row];
                auto & fd = state[i][f];
                if (fd.use_tree) {
                    tree_scores[i][f].push_back(scores[row]);
                } else {
                    const auto & s    = scores[row];
                    const int    best = (int) (std::max_element(s.begin(), s.end()) - s.begin());
                    double denom = 0;
                    for (float x : s) {
                        denom += std::exp((x - s[best]) / fd.temperature);
                    }
                    fd.select(todo[row].cands[best], (float) (1.0 / denom));
                }
            }
            if (first) {
                for (size_t i = 0; i < n_group; ++i) {
                    for (size_t f = 0; f < fields.size(); ++f) {
                        if (state[i][f].use_tree) {
                            state[i][f].finish_tree(tree_scores[i][f]);
                        }
                    }
                }
            }
            first = false;
        }
        for (size_t i = 0; i < n_group; ++i) {
            llama_memory_seq_rm(mem, seq_pool + (llama_seq_id) i, -1, -1);
            result & r = out.items[g0 + i];
            r.context_tokens = contexts[g0 + i].prefill ? contexts[g0 + i].n_tokens : prefixes[g0 + i].size();
            r.rows           = total;
            r.pos_fields     = pos_next[g0 + i];
            r.group          = n_groups;
            for (auto & fd : state[i]) {
                if (fd.use_tree && fd.probs.empty()) {
                    fd.finish_tree({});
                }
                r.fields.push_back({ fd.winner, fd.path_score, fd.scored_nodes, fd.use_tree, fd.probs });
            }
        }
        out.scoring_ms += ms_since(ts);
    }
    guard.armed = false;
    return out;
}

// ---------------------------------------------------------------- schema compiler

namespace {

int decimal_places(double x) {
    for (int k = 0; k <= 9; ++k) {
        const double v = x * std::pow(10.0, k);
        if (std::fabs(v - std::llround(v)) < 1e-9 * std::max(1.0, std::fabs(v))) {
            return k;
        }
    }
    return 9;
}

std::string json_text(const std::string & s) {
    return common_json::make(s).dump();
}

field_spec make_field(const std::string & name, const std::string & type, const std::string & description,
                      const common_json & spec, bool json_schema, bool nullable) {
    field_spec f;
    f.name        = name;
    f.description = description;
    const std::string kind = type;
    if (kind == "boolean") {
        f.type    = "boolean";
        f.values  = { common_json(true), common_json(false) };
        f.encoded = { "true", "false" };
    } else if (kind == "enum" || kind == "choice" || kind == "selection") {
        const char * key = spec.contains("enum") ? "enum" : "choices";
        if (!spec.contains(key) || !spec.at(key).is_array()) {
            throw std::invalid_argument("field \"" + name + "\": enum fields need a list of choices");
        }
        f.type = "enum";
        for (const auto & c : spec.at(key)) {
            if (!c.is_string()) {
                throw std::invalid_argument("field \"" + name + "\": enum choices must be strings");
            }
            const std::string v = c.get<std::string>();
            f.values.push_back(common_json(v));
            f.encoded.push_back(json_text(v));
        }
    } else if (kind == "integer") {
        if (!spec.contains("minimum") || !spec.contains("maximum") ||
            !spec.at("minimum").is_number_integer() || !spec.at("maximum").is_number_integer()) {
            throw std::invalid_argument("field \"" + name + "\": integer fields need integer minimum and maximum");
        }
        const long long lo = spec.at("minimum").get<long long>(), hi = spec.at("maximum").get<long long>();
        if (hi < lo || hi - lo + 1 > 255) {
            throw std::invalid_argument("field \"" + name + "\": integer bounds must define 1-255 values");
        }
        f.type = "integer";
        for (long long v = lo; v <= hi; ++v) {
            f.values.push_back(common_json(v));
            f.encoded.push_back(std::to_string(v));
            f.numbers.push_back((double) v);
        }
    } else if (kind == "number") {
        const char * step_key = json_schema ? "multipleOf" : "step";
        if (!spec.contains("minimum") || !spec.contains("maximum") || !spec.contains(step_key)) {
            throw std::invalid_argument("field \"" + name + "\": number fields need minimum, maximum and " + step_key);
        }
        const double lo = spec.at("minimum").get<double>(), hi = spec.at("maximum").get<double>(),
                     step = spec.at(step_key).get<double>();
        if (!(step > 0) || !(hi >= lo)) {
            throw std::invalid_argument("field \"" + name + "\": number needs ordered bounds and a positive step");
        }
        const double    count = (hi - lo) / step;
        const long long n     = std::llround(count);
        if (std::fabs(count - (double) n) > 1e-7 || n < 0 || n > 254) {
            throw std::invalid_argument("field \"" + name + "\": the number grid must include both ends and hold 1-255 values");
        }
        const int places = std::max({ decimal_places(lo), decimal_places(hi), decimal_places(step) });
        f.type = "number";
        for (long long i = 0; i <= n; ++i) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.*f", places, lo + (double) i * step);
            const double v = std::strtod(buf, nullptr);
            f.values.push_back(common_json(v));
            f.encoded.push_back(buf); // fixed width: every value has the same shape
            f.numbers.push_back(v);
        }
    } else {
        throw std::invalid_argument("field \"" + name + "\": supported types are boolean, enum, integer and number");
    }
    if (nullable) {
        // e.g. a size that the image does not show
        f.nullable = true;
        f.values.push_back(common_json(nullptr));
        f.encoded.push_back("null");
        if (!f.numbers.empty()) {
            f.numbers.push_back(std::nan(""));
        }
    }
    if (f.values.empty() || f.values.size() > 255) {
        throw std::invalid_argument("field \"" + name + "\" needs 1-255 allowed values");
    }
    for (size_t a = 0; a < f.encoded.size(); ++a) {
        for (size_t b = 0; b < a; ++b) {
            if (f.encoded[a] == f.encoded[b]) {
                throw std::invalid_argument("field \"" + name + "\" has duplicate allowed values");
            }
        }
    }
    const std::string agg = spec.value("aggregate", spec.value("x-aggregate", std::string("mode")));
    const bool numeric = f.type == "integer" || f.type == "number";
    if (agg != "mode" && !(numeric && (agg == "median" || agg == "mean"))) {
        throw std::invalid_argument("field \"" + name + "\": aggregate must be mode, or median/mean for numeric fields");
    }
    f.aggregate = agg;
    return f;
}

} // namespace

compiled_schema compile_schema(const common_json & schema, const std::string & instructions, const common_json & defaults) {
    if (!schema.is_object()) {
        throw std::invalid_argument("\"schema\" must be an object");
    }
    compiled_schema cs;
    const bool json_schema = schema.contains("properties");
    const common_json & props = json_schema ? schema.at("properties") : schema;
    if (!props.is_object() || props.size() < 1 || props.size() > 32) {
        throw std::invalid_argument("the schema must define 1-32 fields");
    }
    for (const auto & e : props.items()) {
        const common_json & spec = e.value();
        if (!spec.is_object()) {
            throw std::invalid_argument("field \"" + e.key() + "\" must be an object");
        }
        // null can be allowed with "nullable": true, a "null" in a type list, or a null in an enum
        bool nullable = spec.value("nullable", false);
        std::string type;
        if (spec.contains("type") && spec.at("type").is_array()) {
            for (const auto & t : spec.at("type")) {
                if (t == "null") {
                    nullable = true;
                } else if (t.is_string()) {
                    type = t.get<std::string>();
                }
            }
        } else {
            type = spec.value("type", std::string());
        }
        common_json field = spec;
        if (field.contains("enum")) {
            type = "enum";
            common_json choices = common_json::array();
            for (const auto & c : field.at("enum")) {
                if (c.is_null()) {
                    nullable = true;
                } else {
                    choices.push_back(c);
                }
            }
            field["enum"] = choices;
        }
        std::string description = spec.value("description", std::string());
        if (!json_schema && description.empty()) {
            throw std::invalid_argument("field \"" + e.key() + "\" needs a description");
        }
        field_spec f = make_field(e.key(), type, description, field, json_schema, nullable);
        const char * t_key = json_schema ? "x-temperature" : "temperature";
        const char * a_key = json_schema ? "x-abstain" : "abstain";
        const common_json t = spec.contains(t_key) ? spec.at(t_key) : defaults.value("temperature", common_json(1.0));
        if (!t.is_number() || !(t.get<double>() > 0) || t.get<double>() > 100) {
            throw std::invalid_argument("field \"" + f.name + "\": temperature must be a number in (0, 100]");
        }
        f.temperature = t.get<float>();
        const common_json ab = spec.contains(a_key) ? spec.at(a_key) : defaults.value("abstain", common_json::object());
        if (!ab.is_object()) {
            throw std::invalid_argument("field \"" + f.name + "\": abstain must be an object");
        }
        f.min_probability = ab.value("min_probability", 0.0);
        f.min_margin      = ab.value("min_margin", 0.0);
        if (f.min_probability < 0 || f.min_probability > 1 || f.min_margin < 0 || f.min_margin > 1) {
            throw std::invalid_argument("field \"" + f.name + "\": abstain thresholds must be in [0, 1]");
        }
        cs.specs.push_back(std::move(f));
    }

    std::string catalog;
    for (const auto & f : cs.specs) {
        // the value's common leading characters are fixed in the suffix; only the rest is scored
        std::string common = f.encoded[0];
        for (const auto & v : f.encoded) {
            size_t c = 0;
            while (c < std::min(common.size(), v.size()) && common[c] == v[c]) {
                ++c;
            }
            common.resize(c);
        }
        field_input in;
        in.suffix      = "  " + json_text(f.name) + ": " + common;
        in.temperature = f.temperature;
        for (const auto & v : f.encoded) {
            in.candidates.push_back(v.substr(common.size()));
        }
        cs.inputs.push_back(in);

        std::string allowed;
        for (size_t i = 0; i < f.encoded.size(); ++i) {
            allowed += (i ? ", " : "") + f.encoded[i];
        }
        catalog += (catalog.empty() ? "" : "\n") + json_text(f.name) + (f.description.empty() ? "" : ": " + f.description) +
                   "\nAllowed values: " + allowed;
    }
    cs.system_text = "Select the requested field value from its allowed values, based on the context. "
                     "Respond with the JSON value only.\n\nFields:\n" + catalog + "\n" + instructions;
    return cs;
}

std::pair<std::string, std::string> render_prompt(const common_chat_templates * tmpls, bool use_jinja,
                                                  const std::string & system_text, const std::string & context) {
    if (tmpls == nullptr) {
        return { system_text + "\nContext:\n", context + "\nOutput:\n{\n" };
    }
    static const std::string sentinel = "\x1f<<decision-context>>\x1f";
    common_chat_templates_inputs in;
    in.use_jinja             = use_jinja;
    in.add_generation_prompt = true;
    in.enable_thinking       = false;
    common_chat_msg sys;
    sys.role    = "system";
    sys.content = system_text;
    common_chat_msg usr;
    usr.role    = "user";
    usr.content = sentinel;
    in.messages = { sys, usr };
    const std::string prompt = common_chat_templates_apply(tmpls, in).prompt;
    const size_t at = prompt.find(sentinel);
    if (at == std::string::npos) {
        throw std::runtime_error("the chat template did not keep the user message");
    }
    return { prompt.substr(0, at), context + prompt.substr(at + sentinel.size()) + "{\n" };
}

common_json assemble(const compiled_schema & cs, const result & r, bool with_probs) {
    common_json decision  = common_json::object();
    common_json fields    = common_json::object();
    common_json abstained = common_json::array();
    bool        rules     = false;
    for (size_t i = 0; i < cs.specs.size(); ++i) {
        const auto & sp = cs.specs[i];
        const auto & fr = r.fields[i];
        int idx = fr.winner;
        common_json f = common_json::object();
        const bool numeric = !sp.numbers.empty();
        const bool tree    = fr.probs.size() == sp.values.size();
        if (numeric && tree) {
            // aggregates and the interval use the numbers only; a nullable field is null when p(null) >= 0.5
            int null_idx = -1;
            std::vector<int> order;
            for (size_t k = 0; k < sp.numbers.size(); ++k) {
                if (std::isnan(sp.numbers[k])) {
                    null_idx = (int) k;
                } else {
                    order.push_back((int) k);
                }
            }
            const double p_num = 1.0 - (null_idx >= 0 ? fr.probs[null_idx] : 0.0);
            std::sort(order.begin(), order.end(), [&](int a, int b) { return sp.numbers[a] < sp.numbers[b]; });
            auto quantile = [&](double q) {
                double acc = 0;
                for (int k : order) {
                    acc += fr.probs[k] / p_num;
                    if (acc >= q) {
                        return k;
                    }
                }
                return order.back();
            };
            if (null_idx >= 0 && fr.probs[null_idx] >= 0.5) {
                idx = null_idx;
            } else if (p_num > 0 && !order.empty()) {
                double mean = 0;
                for (int k : order) {
                    mean += sp.numbers[k] * fr.probs[k] / p_num;
                }
                if (sp.aggregate == "median") {
                    idx = quantile(0.5);
                } else if (sp.aggregate == "mean") {
                    idx = order[0];
                    for (int k : order) {
                        if (std::fabs(sp.numbers[k] - mean) < std::fabs(sp.numbers[idx] - mean)) {
                            idx = k;
                        }
                    }
                } else if (idx == null_idx) {
                    idx = *std::max_element(order.begin(), order.end(), [&](int a, int b) { return fr.probs[a] < fr.probs[b]; });
                }
                common_json interval = common_json::array();
                interval.push_back(sp.values[quantile(0.1)]);
                interval.push_back(sp.values[quantile(0.9)]);
                f["interval_p10_p90"] = interval;
            }
        }
        if (idx < 0 || idx >= (int) sp.values.size()) {
            throw std::runtime_error("field \"" + sp.name + "\" has no selected value");
        }
        const double p = tree ? fr.probs[idx] : fr.path_score;
        decision[sp.name] = sp.values[idx];
        f["value"]        = sp.values[idx];
        f["probability"]  = p;
        f["scored_nodes"] = fr.scored_nodes;
        f["tree"]         = fr.tree;
        double margin = 1.0;
        if (tree) {
            std::vector<float> sorted = fr.probs;
            std::sort(sorted.begin(), sorted.end(), std::greater<float>());
            margin = sorted[0] - (sorted.size() > 1 ? sorted[1] : 0.0f);
            double entropy = 0;
            for (float q : fr.probs) {
                entropy -= q > 0 ? q * std::log(q) : 0.0;
            }
            f["margin"]  = margin;
            f["entropy"] = entropy; // nats
            if (with_probs) {
                common_json probs = common_json::array();
                for (size_t k = 0; k < sp.values.size(); ++k) {
                    probs.push_back({ { "value", sp.values[k] }, { "probability", fr.probs[k] } });
                }
                f["probs"] = probs;
            }
        }
        if (sp.min_probability > 0 || sp.min_margin > 0) {
            rules = true;
            const bool abstain = p < sp.min_probability || (tree && margin < sp.min_margin);
            f["abstain"] = abstain;
            if (abstain) {
                abstained.push_back(sp.name);
            }
        }
        fields[sp.name] = f;
    }
    common_json out = common_json::object();
    out["decision"] = decision;
    out["fields"]   = fields;
    if (rules) {
        out["abstained"] = abstained;
    }
    return out;
}
} // namespace llama_decision
