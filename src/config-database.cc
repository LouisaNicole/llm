#include "config-database.hh"
#include <unordered_map>
#include <cassert>

static std::unordered_map<std::string, Config*> configs;
static bool db_inited;
const Config *g_config;

void load_config(const std::string &model, const std::string &llm, bool skip_l1_hit) {
    if (!db_inited) {
        configs["Intel 13900K"] = new ConfigIntel13900K(llm, skip_l1_hit);
        configs["Intel 14900K"] = new ConfigIntel14900K(llm, skip_l1_hit);
        configs["Intel 12700KF"] = new ConfigIntel12700KF(llm, skip_l1_hit);
    }
    assert(configs.find(model) != configs.end());
    g_config = configs[model];
}
