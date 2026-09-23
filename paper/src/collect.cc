#include <string>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <filesystem>
#include <vector>
#include <string>
#include <functional>
#include <atomic>
#include <set>
#include <random>
#include <chrono>
#include "config-database.hh"
#include "util.h"
#include "attack.hh"
#include "tokenizer.hh"
#include "progressbar.hh"

#include <signal.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

std::atomic<bool> stopped;
std::string hw, llm, device, prog_pathname, model_file, model_type, dataset_pathname, result_database;
size_t max_out_tokens;
bool set_prefill_bs;
std::string prefill_bs, embd_quant;

static void copy_file(const std::string &from, const std::string &to_dir) {
    auto to = to_dir + "/" + from;
    if (!std::filesystem::copy_file(from, to)) {
        std::cerr << "failed on copying '" << from << "' to '" << to << "'" << std::endl;
        exit(-1);
    }
}

static std::string get_cur_datetime() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm local_time = *std::localtime(&now_time_t);
    std::stringstream oss;
    oss << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S"); 
    return oss.str();
}

bool is_instruct(const std::string &llm) {
    if (!llm.compare("llama-2-7b-chat"))
        return true;
    assert(0);
    return false;
}

static void signal_stop_handler(int signum) {
    if (signum == SIGINT) {
        stopped.store(true);
        std::cout << "Request to stop..." << std::endl;
    }
}

static int run_llamacpp(Attacker *attack, const std::string &user_prompt) {    
    std::string promot_filename = get_temp_dir();
    promot_filename += "/.tmp.txt";
    std::ofstream fprompt(promot_filename);
    if (!fprompt)
        return -1;
    // apply chat template
    fprompt << cfg().chat_template(user_prompt);
    fprompt.close();

    auto max_out_tokens_s = tostr(max_out_tokens);
    auto ngl_s = tostr(cfg().n_gpu_layers);
	std::vector<const char *> proc_argv_list = {
        prog_pathname.c_str(),
        "-m", model_file.c_str(),
        "-f", promot_filename.c_str(),
        "-s", "123", 
        "--max-out-tokens", max_out_tokens_s.c_str(),
        "--ctx-size", "4096",
        "--repeat-penalty", "1.01", // ensure that the output will be always terminated
    };
    if (!device.compare("gpu")) {
        proc_argv_list.push_back("--n-gpu-layers");
        proc_argv_list.push_back(ngl_s.c_str());

    } else if (!device.compare("cpu")) {
        // nothing to be done
    } else {
        std::cerr << "Unknown device " << device << std::endl;
        assert(0);
    }
	if (set_prefill_bs) {
		proc_argv_list.push_back("-b");
		proc_argv_list.push_back(prefill_bs.c_str());
	}
#if 0 
    unsigned N=proc_argv_list.size();
    std::stringstream ss;
    for(unsigned i=0;i<N;++i) {
        ss<<proc_argv_list[i]<<" ";
    }
    std::cout<<"cmdline: " << ss.str() << std::endl;
#endif

    auto proc = create_process(proc_argv_list.data(), proc_argv_list.size(), true);
    if (!proc) {
        printf("create_process()\n");
        return -1;
    }
    int status;
    attack->loop([&]() -> int {
        if (wait_process(proc, status)) {
            free_process(proc);
            return -1;
        }
        return 0;
    }, /* echo */ 0);
    if (status) {
        std::cerr << model_type << " exited with code " << status << std::endl;
    }
    return status;
}

std::string get_expriment_name() {
	std::stringstream bs_suffix;
	if (set_prefill_bs)
		bs_suffix << "_bs" << prefill_bs;
    if (embd_quant != "F16")
        bs_suffix << "_embd" << embd_quant;
    return llm + "_" + std::filesystem::path(dataset_pathname).stem().string() + bs_suffix.str();
}

bool parse_inc_opt(const char *v) {
    if (!strcmp(v,"resume"))
        return true;
    else {
        if (strcmp(v,"new")) {
            std::cerr << "invalid value " << v << std::endl;
            exit(1);
        }
        return false;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 15) {
        printf("Usage %s hw\n"
                "llm\n"
                "device\n"
                "prog_pathname\n"
                "model_file\n"
                "model_type\n"
                "dataset_pathname\n"
                "result_database\n"
                "max_n_prompt_tokens\n"
                "incremental\n"
                "checkpoint_freq\n"
                "max_out_tokens\n"
                "prefill_batch_size\n"
                "embd_quant\n"
                "<skip_l1_hit>\n",
                argv[0]);
        return 1;
    }

    hw = argv[1],
    llm = argv[2],
    device = argv[3],
    prog_pathname = argv[4],
    model_file = argv[5],
    model_type = argv[6],
    dataset_pathname = argv[7],
    result_database = argv[8];
    size_t max_n_prompt_tokens = strtol(argv[9], nullptr, 10);
    auto incremental = parse_inc_opt(argv[10]);
    size_t checkpoint_freq = strtol(argv[11], nullptr, 10);
    max_out_tokens =  strtol(argv[12], nullptr, 10); 
	prefill_bs = argv[13];
    set_prefill_bs = strcmp(prefill_bs.c_str(), "default");
    embd_quant = argv[14];

    bool skip_l1_hit = false;
    if (argc == 16) {
        skip_l1_hit = true;
    } else if (argc > 16) {
        printf("Too many args\n");
        return 1;
    }

	printf("set_prefill_batch_size = %d\n", set_prefill_bs);
	if (set_prefill_bs)
		printf("prefill_batch_size = %s\n", prefill_bs.c_str());

    load_config(hw, llm, skip_l1_hit);

    assert(PAGE_SIZE == get_page_size());

    srand(1023);

    auto attack = std::make_unique<Attacker>();
    int ret = attack->init(model_file, model_type);
    if (ret)
        return ret;

    // mkdir
    auto save_dir = result_database + "/" + hw;
    std::filesystem::create_directories(save_dir);
    
    // load datasets
    json ds;
    std::ifstream fin_dataset(dataset_pathname);
    assert(fin_dataset);
    fin_dataset >> ds;
    auto n_total_prompts = 0;
    for (auto &ds : ds["datasets"]) {
        n_total_prompts += ds["samples"].size();
    }

    // check the existing results
    std::string database_file = save_dir + "/" + get_expriment_name() + ".json";
    json db_obj;
    std::set<std::tuple<size_t, size_t>> completed; // <dataset_id, sample_id>
    if (std::filesystem::exists(database_file)) {
        if (incremental) {
            // check the consistency of dataset
            std::ifstream fmon_in(database_file);
            assert(fmon_in);
            fmon_in >> db_obj;
            try {
#if 0 // unused
                if (db_obj["seed"] != ds["seed"]) {
                    std::cerr << "ERROR: seed is mismatched with previous experiments." << std::endl;
                    return 1;
                }
                if (db_obj["prompt_hash"] != ds["prompt_hash"]) {
                    std::cerr << "ERROR: prompt_hash is mismatched with previous experiments." << std::endl;
                    return 1;
                }
#endif
                std::cout << " *** NOTE: incremental appending to the previous results ***" << std::endl;
                for (auto &item : db_obj["results"]) {
                    completed.insert(std::make_tuple(item["d"], item["s"]));
                }
            } catch(std::exception &e) {
                std::cerr << e.what() << std::endl;
                return 1;
            }
        } else {
            std::cout << " !!! override the previous results !!!" << std::endl;
        }
    }

#if 0
    db_obj["seed"] = ds["seed"];
    db_obj["prompt_hash"] = ds["prompt_hash"];
#endif
    auto &results = db_obj["results"];
    
    auto save_ds_obj = [&]() {
        std::ofstream fout(database_file);
        assert(fout);
        fout << db_obj;
        std::cout << "written to '" << database_file << "'" << std::endl;
    };
    
    stopped.store(false);
    signal(SIGINT, &signal_stop_handler);

    ProgressBar<size_t> pbar(n_total_prompts, 1);
    size_t n_tested_prompts = 0;
    int status;

    // warming up the machine to achieve relatively stable temperature/computing performance
    if ((status = run_llamacpp(attack.get(), "say nothing")))
        return -1;
    std::cout << "Running..." << std::endl;

    std::ofstream flog(result_database + "/" + hw + "/" + get_expriment_name() + ".log",
                    incremental ? std::ofstream::app : std::ofstream::out);
    // logging timestamp
    flog << "Logged Date and Time: " << get_cur_datetime() << std::endl;

    flog << "Out cache trace results: " << database_file << std::endl;

    if (!incremental) {
        flog << "checkpoint_freq = " << checkpoint_freq << std::endl
            << "max_out_tokens = " << max_out_tokens << std::endl;
    }

    size_t checkpoint = 0;

    auto &datasets = ds["datasets"];
    for(size_t k = 0; k < datasets.size(); ++k) {
        std::cout << "Evaluating dataset " << datasets[k]["name"] << std::endl;
        
        auto &samples = datasets[k]["samples"];
        for (size_t i = 0; i < samples.size(); ++i) {
            auto &prompt = samples[i]["prompt"];
            
            // incremental
            if (completed.find(std::make_tuple(k,i)) != completed.end()) {
                std::cout << "skipped " << k << " " << i << std::endl;
                pbar.drop_total(1);
                continue;
            }
            if ((status = run_llamacpp(attack.get(), prompt)))
                goto out;

            // read the ground truth
            json groundtruth_json;
            std::ifstream fgt("groundtruth.json");
            fgt >> groundtruth_json;
            std::vector<llama_token> groundtruth;
            for(auto &tk : groundtruth_json["i"]) {
                assert(tk.is_number());
                unsigned token;
                groundtruth.emplace_back(tk.get_to(token));
            }
            for(auto &tk : groundtruth_json["o"]) {
                assert(tk.is_number());
                unsigned token;
                groundtruth.emplace_back(tk.get_to(token));
            }
            
            auto sorted_detected = attack->get_detected();

            // compute the metrics for preview
            std::unordered_multiset<unsigned> detected_set;
            std::unordered_set<unsigned> groundtruth_set;
            for (auto &e : sorted_detected)
                detected_set.insert(e.row);
            for (auto &tk : groundtruth) {
                groundtruth_set.insert(tk);
            }
            double precision, recall;
            size_t mon_tp=0, mon_fp=0, mon_fn=0;
            auto tmp_set = detected_set;
            for(auto &token : groundtruth) {
                auto it = tmp_set.find(token);
                if (it != tmp_set.end()) {
                    tmp_set.erase(it);
                    mon_tp++;
                } else {
                    mon_fn++;
                }
            }
            for (auto tk : detected_set) {
                if (groundtruth_set.find(tk) == groundtruth_set.end())
                    mon_fp++;
            }
            precision = double(mon_tp)/(mon_tp+mon_fp);
            recall = double(mon_tp)/(mon_tp+mon_fn);
            
            
            flog << "+" << k << " " << i << std::endl;
            flog << "p: " << prompt << std::endl;
            flog << "r: ";
            char buf[256];
            for (auto &e : sorted_detected) {
                int ret = llama_token_to_piece(attack->num_vocab, attack->vocab, e.row, buf, sizeof(buf));
                if (ret < 0)
                    strcpy(buf, "(err)");
                else
                    buf[ret] = '\0';
                flog << buf;
            }
            flog << std::endl;
            
            // append db_obj
            auto events_obj = json::array();
            for(auto &res : sorted_detected) {
                json res_obj = {res.row, res.latency, res.timestamp};
                events_obj.emplace_back(res_obj);
            }
            // inhert the keys
            json item = samples[i];
            item["ds_name"] = datasets[k]["name"];
            // new keys
            item["d"] = k;
            item["s"] = i;
            item["e"] = events_obj;
            item["g"] = groundtruth_json;
            item["probe_pr"] = precision;
            item["probe_rc"] = recall;
            results.emplace_back(item);

            pbar.update(n_tested_prompts++, false);
            std::cout << " | precision: " << precision * 100 << "% recall: " << recall * 100 << "%" << std::flush;
            flog << "precision: " << precision * 100 << "% recall: " << recall * 100 << "%" << std::endl;

            if (stopped) {
                goto out;
            }

            if (++checkpoint == checkpoint_freq) {
                checkpoint = 0;
                save_ds_obj();
            }
        }
    }

    pbar.end();
    //flog << "elapsed time: " << pbar.get_elapsed_time() << "sec" << std::endl;
    flog << "Finished at Date and Time: " << get_cur_datetime() << std::endl;

out:
    save_ds_obj();
    if (status)
        std::cerr << "status = " << status << std::endl;
    if (stopped) {
        std::cerr << "(canceled)\n";
        flog << "Canceled at Date and Time: " << get_cur_datetime() << std::endl;
    }
    return stopped ? 1 : status;
}
