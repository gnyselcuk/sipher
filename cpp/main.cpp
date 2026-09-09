// Sipher FHE Inference — CLI
#include "sipher_fhe.h"
#include <sstream>

void print_usage(const char* prog) {
    std::cout << "Sipher FHE Inference Engine\n"
              << "Usage:\n"
              << "  " << prog << " --weights <dir> --prompt \"text\" [--layers N] [--temp 0.7]\n"
              << "  " << prog << " --weights <dir> --benchmark [--d 32]\n"
              << "\nOptions:\n"
              << "  --weights <dir>   Weight directory (from export_weights.py)\n"
              << "  --prompt \"text\"   Input prompt\n"
              << "  --layers N        Number of layers to run (default: 1)\n"
              << "  --temp T          Sampling temperature (default: 0.7)\n"
              << "  --top-k K         Top-k sampling (default: 40)\n"
              << "  --benchmark       Run FHE benchmark\n"
              << "  --d D             Dimension for benchmark (default: 32)\n";
}

int main(int argc, char* argv[]) {
    std::string weights_dir = "weights";
    std::string prompt = "";
    int n_layers = 1;
    double temp = 0.7;
    int top_k = 40;
    bool benchmark = false;
    int bench_d = 32;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--weights" && i + 1 < argc) weights_dir = argv[++i];
        else if (arg == "--prompt" && i + 1 < argc) prompt = argv[++i];
        else if (arg == "--layers" && i + 1 < argc) n_layers = std::stoi(argv[++i]);
        else if (arg == "--temp" && i + 1 < argc) temp = std::stod(argv[++i]);
        else if (arg == "--top-k" && i + 1 < argc) top_k = std::stoi(argv[++i]);
        else if (arg == "--benchmark") benchmark = true;
        else if (arg == "--d" && i + 1 < argc) bench_d = std::stoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
    }

    std::cout << "╔══════════════════════════════════════════════════════════╗\n"
              << "║  Sipher FHE Inference Engine (OpenFHE CKKS)            ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n\n";

    // Init CKKS
    CKKSContext ckks;
    std::cout << "CKKS context initializing...\n";
    Timer init_timer;
    ckks.init(65536, 40, 20);
    std::cout << "  Init: " << init_timer.elapsed() << "s\n\n";

    if (benchmark) {
        // Benchmark: matvec at different dimensions
        std::cout << "═══ BENCHMARK ═══\n";
        for (int d : {32, 64, 128, 256, 512, 1024}) {
            if (d > bench_d * 32) break;

            std::vector<double> x(d);
            for (int i = 0; i < d; i++) x[i] = ((double)rand() / RAND_MAX - 0.5) * 0.6;

            Tensor W;
            W.shape = {d, d};
            W.data.resize(d * d);
            for (int i = 0; i < d * d; i++)
                W.data[i] = ((double)rand() / RAND_MAX - 0.5) * 0.6;

            auto enc_x = ckks.encrypt(x);

            Timer t;
            auto enc_y = ckks.matvec(enc_x, W, d, d);
            double elapsed = t.elapsed();

            auto y_fhe = ckks.decrypt(enc_y, d);

            // Plaintext reference
            std::vector<double> y_pt(d, 0.0);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    y_pt[i] += W.at(i, j) * x[j];

            double max_err = 0.0;
            for (int i = 0; i < d; i++)
                max_err = std::max(max_err, std::abs(y_pt[i] - y_fhe[i]));

            std::cout << "  d=" << d << "  matvec: " << elapsed << "s  max_err=" << max_err << "\n";
        }

        // ct×ct benchmark
        std::cout << "\n  ct×ct benchmark:\n";
        for (int d : {32, 64, 128, 256}) {
            std::vector<double> x(d, 0.5);
            auto enc_x = ckks.encrypt(x);
            Timer t;
            auto enc_x2 = ckks.mult(enc_x, enc_x);
            double elapsed = t.elapsed();
            auto x2 = ckks.decrypt(enc_x2, d);
            double err = std::abs(x2[0] - 0.25);
            std::cout << "  d=" << d << "  ct×ct: " << elapsed << "s  err=" << err << "\n";
        }
        return 0;
    }

    // Load model
    std::cout << "Model loading from: " << weights_dir << "\n";
    SipherModel model;
    model.load(weights_dir);

    if (prompt.empty()) {
        prompt = "Merkez Bankası Başkanı bugün yaptığı açıklamada";
        std::cout << "  Default prompt: \"" << prompt << "\"\n";
    }

    // Simple tokenization (char-level for PoC)
    // In production, use the Sipher BPE tokenizer
    std::vector<int> tokens;
    for (char c : prompt) {
        tokens.push_back((int)(unsigned char)c % model.config.vocab_size);
    }
    // Pad to seq_len
    while ((int)tokens.size() < model.config.seq_len) {
        tokens.insert(tokens.begin(), 0);
    }

    std::cout << "\n═══ FHE INFERENCE ═══\n";
    std::cout << "  Prompt: \"" << prompt << "\"\n";
    std::cout << "  Tokens: " << tokens.size() << "\n";
    std::cout << "  Layers: " << n_layers << " / " << model.config.n_layers << "\n\n";

    Timer total_timer;
    int next = model.generate_next(ckks, tokens, temp, top_k);
    double total_time = total_timer.elapsed();

    int actual_layers = model.config.n_layers;
    std::cout << "\n═══ RESULT ═══\n";
    std::cout << "  Next token ID: " << next << "\n";
    std::cout << "  Total time: " << total_time << "s\n";
    std::cout << "  Layers: " << actual_layers << "\n";
    std::cout << "  Per-layer avg: " << total_time / actual_layers << "s\n";
    std::cout << "  Tokens/s: " << 1.0 / total_time << "\n";
    std::cout << "  With GPU (50x): " << 1.0 / (total_time / 50) << " tokens/s\n";
    std::cout << "  With ASIC (1000x): " << 1.0 / (total_time / 1000) << " tokens/s\n";

    return 0;
}
