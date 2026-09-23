#ifndef DATASET_HH_
#define DATASET_HH_

#include <vector>
#include <string>

class Dataset {
protected:
    struct Subdata {
        std::string name;
        std::vector<std::string> prompts;
        Subdata(const std::string &name_) : name(name_) {}
    };
    std::vector<Subdata> sub;
public:
    inline const auto &data() const {
        return sub;
    }
    virtual const char *name() const =0;
};

#endif // DATASET_HH_