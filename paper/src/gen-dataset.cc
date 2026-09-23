#include <iostream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#include <rapidcsv.h>
#include <unordered_map>
#include <random>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>

#include "dataset.hh"
#include "tokenizer.hh"

static void enum_subdirs(const std::filesystem::path& directory, const std::function<void(const std::filesystem::path& fn)> &callback) {
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (std::filesystem::is_directory(entry.status())) {
            enum_subdirs(entry.path(), callback);
        } else if (std::filesystem::is_regular_file(entry.status())) {
           callback(entry.path());
        } 
    }
}

static bool is_jsonl(const std::string &fn) {
    return (fn.substr(fn.size()-6) == ".jsonl");
}


class DatasetNQ_Open : public Dataset {
public:
    DatasetNQ_Open(const std::string &rootpath) {
        sub.emplace_back("nq-open");
        Subdata &s = *std::prev(sub.end(),1);

        std::ifstream fin(rootpath + "/nq_open/NQ-open.train.jsonl"); 
        assert(fin);
        std::string line;
        while(std::getline(fin, line)) {
            auto obj = json::parse(line);
            s.prompts.emplace_back(std::string(obj["question"]) + "?");
        }
    }
    const char *name() const override { return "nq-open"; }
};

class DatasetChatGPT_Roles : public Dataset {
public:
    DatasetChatGPT_Roles(const std::string &rootpath) {
        sub.emplace_back("chatgpt-roles");
        Subdata &s = *std::prev(sub.end(),1);

        std::ifstream fin(rootpath + "/train.jsonl");
        assert(fin);
        std::string line;
        while(std::getline(fin, line)) {
            auto obj = json::parse(line);
            s.prompts.emplace_back(std::string(obj["prompt"]) + "?");
        }
    }
    const char *name() const override { return "chatgpt-roles"; }
};

class DatasetSIQA : public Dataset {
public:
    DatasetSIQA(const std::string &rootpath) {
        sub.emplace_back("SIQA");
        Subdata &s = *std::prev(sub.end(),1);

        std::ifstream fin(rootpath + "/train.jsonl");
        assert(fin);
        std::string line;
        while(std::getline(fin, line)) {
            auto obj = json::parse(line);
            s.prompts.emplace_back(std::string(obj["prompt"]) + "?");
        }
    }
    const char *name() const override { return "SIQA"; }
};

class DatasetSQuAD2 : public Dataset {
public:
    DatasetSQuAD2(const std::string &rootpath) {
        sub.emplace_back("SQuAD2");
        Subdata &s = *std::prev(sub.end(),1);

        std::ifstream fin(rootpath + "/train.jsonl");
        assert(fin);
        std::string line;
        while(std::getline(fin, line)) {
            auto obj = json::parse(line);
            s.prompts.emplace_back(std::string(obj["prompt"]) + "?");
        }
    }
    const char *name() const override { return "SQuAD2"; }
};

class DatasetUltraChat : public Dataset {
   
public:
    DatasetUltraChat(const std::string &rootpath) {
        std::vector<std::pair<std::string,std::string>> fns;
        fns.emplace_back(std::make_pair("QA_W", rootpath + "/ultrachat_release_230407.json"));
        
        for (size_t i = 0; i < fns.size(); ++i) {
            sub.emplace_back(fns[i].first);
            auto &ds = sub.back();
            std::ifstream fin(fns[i].second);
            std::cout << "loading " << fns[i].second <<std::endl;
            assert(fin);
            std::string line;
            while(std::getline(fin, line)) {
                auto obj = json::parse(line);
                auto id = obj["id"];
                ds.prompts.emplace_back(obj["data"][0]); /* used the first prompt only to form our dataset  */
            }
        }
    }
    const char *name() const override { return "UltraChat"; }
};


int main(int argc, char *argv[]) {
    if (argc < 7) {
        printf("Usage %s dataset_dir out_dataset max_n_prompt_tokens ref_tokenizer_weight_file seed collection train/test\n", argv[0]);
        return 1;
    }
    std::string dataset_dir = argv[1],
                out_pathname = argv[2];
    size_t max_n_prompt_tokens = strtol(argv[3], nullptr, 10);
    auto ref_tokenizer_weight_file = argv[4];
    auto seed = strtol(argv[5], nullptr, 10);
    std::string collection = argv[6];

    std::vector<Dataset *> ds;
    if (collection == "natural-language") {
        ds.push_back(new DatasetUltraChat(dataset_dir + "/ultrachat"));
        ds.push_back(new DatasetNQ_Open(dataset_dir + "/natural-questions"));
        ds.push_back(new DatasetChatGPT_Roles(dataset_dir + "/ChatGPT-Roles"));
        ds.push_back(new DatasetSIQA(dataset_dir + "/SIQA"));
        ds.push_back(new DatasetSQuAD2(dataset_dir + "/SQuAD2"));
    } else {
        std::cerr << "Invalid collection '" << collection << "'" << std::endl;
        return 1;
    }
    std::cout << "=== Dataset Overview === " << std::endl;
    for(auto &d : ds) {
        std::cout << "Dataset " << d->name() << std::endl;
        for (auto s : d->data()) {
            std::cout << " - " << s.name << ", " << s.prompts.size() << " prompts" << std::endl;
        }
    }

    // load the reference tokenizer
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
    };
    struct gguf_context *ctx_gguf = gguf_init_from_file(ref_tokenizer_weight_file, params);
    if (!ctx_gguf) {
        printf("%s: failed to load model from %s\n", __func__, ref_tokenizer_weight_file);
        return 1;
    }
    llm_arch arch;
    llama_vocab vocab;
    llm_load_arch(ctx_gguf, arch);
    auto ret = llm_load_vocab(ctx_gguf, vocab, arch);
    if (ret) {
        printf("failed\n");
        return ret;
    }

    // randomly sample the dataset with limtiation of token count
    std::mt19937 rng(seed);
    size_t n_prompt_tokens = 0;
    std::vector<std::vector<std::vector<size_t>>> perm;
    std::vector<std::vector<size_t>> indices;
    std::vector<std::vector<std::vector<std::string>>> sampled_ds;
    perm.resize(ds.size());
    indices.resize(ds.size());
    for(size_t k = 0; k < ds.size(); ++k) {
        auto N=ds[k]->data().size();
        perm[k].resize(N);
        indices[k].resize(N);
        for (size_t i = 0; i < N; ++i) {
            auto n_p = ds[k]->data()[i].prompts.size();
            perm[k][i].resize(n_p);
            for (size_t p = 0; p < n_p; ++p) {
                perm[k][i][p] = p;
            }
            std::shuffle(perm[k][i].begin(), perm[k][i].end(), rng);
            indices[k][i] = 0;
        }
    }
    
    for(size_t k = 0; k < ds.size(); ++k) {
        auto &d = ds[k];
        size_t n_prompt_tokens_per_ds = 0, coverged_subs = 0;
        do {
            bool selected_one = false;
            for (size_t i = 0; i < d->data().size(); ++i) { // foreach sub dataset
                if (coverged_subs == ds[k]->data().size() && n_prompt_tokens_per_ds >= max_n_prompt_tokens)
                    break;
                auto &s = d->data()[i];

                // randomly choose a prompt
                if (indices[k][i] == s.prompts.size())
                    continue;
                size_t sel = perm[k][i][indices[k][i]++];
                selected_one = true;

                if (indices[k][i] == 1)
                    ++coverged_subs;

                auto prompt = s.prompts[sel];
                if (k >= sampled_ds.size())
                    sampled_ds.resize(k + 1);
                if (i >= sampled_ds[k].size())
                    sampled_ds[k].resize(i + 1);
                sampled_ds[k][i].emplace_back(prompt);

                auto tokens = tokenize(vocab, prompt, false, false);
                n_prompt_tokens_per_ds += tokens.size();
            }
            if (!selected_one)
                break; // all samples have been selected in this dataset
        } while(coverged_subs != ds[k]->data().size() || n_prompt_tokens_per_ds < max_n_prompt_tokens);
        std::cout  << " Sampled " << n_prompt_tokens_per_ds << " tokens from dataset " << ds[k]->name() << std::endl;
        n_prompt_tokens += n_prompt_tokens_per_ds;
    }
    
    size_t n_total_prompts = 0;
    std::cout << "== Sampled Dataset Overview ==" << std::endl;
    for(size_t k = 0; k < sampled_ds.size(); ++k) {
        std::cout << "Dataset " << ds[k]->name() << std::endl;
        for (size_t i = 0; i < sampled_ds[k].size(); ++i) {
            std::cout << " - " << ds[k]->data()[i].name << ", " << sampled_ds[k][i].size() << " prompts" << std::endl;
            n_total_prompts += sampled_ds[k][i].size();
        }
    }
    std::cout << "Number of sampled prompt tokens = " << n_prompt_tokens << std::endl;

    // save the sampled dataset
    json ds_obj;
    ds_obj["seed"] = seed;
    
    auto sampled_ds_obj = json::array();
    for(size_t k = 0; k < sampled_ds.size(); ++k) {
        json ds_obj;
        ds_obj["name"] = ds[k]->name();
        ds_obj["samples"] = json::array();

        for (size_t i = 0; i < sampled_ds[k].size(); ++i) {
            for(const auto &prompt : sampled_ds[k][i]) {
                json smpl_obj;
                smpl_obj["ds_sub"] = ds[k]->data()[i].name;
                smpl_obj["prompt"] = prompt;
                ds_obj["samples"].emplace_back(smpl_obj);
            }
        }
        sampled_ds_obj.emplace_back(ds_obj);
    }
    ds_obj["datasets"] = sampled_ds_obj;

    ds_obj["n_prompt_tokens"] = n_prompt_tokens;
    ds_obj["n_total_prompts"] = n_total_prompts;
    
    std::ofstream fds(out_pathname);
    assert(fds);
    fds << ds_obj;
    std::cout <<"Written to " << out_pathname << std::endl;

    return 0;
}