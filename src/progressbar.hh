#ifndef PROGRESSBAR_HH_
#define PROGRESSBAR_HH_

#include <iostream>
#include <iomanip>
#include <chrono>

template <typename Tp>
class ProgressBar {
    Tp total;
    unsigned prt_cnt{0}, refresh, ncols;
    bool inited{false};
    std::chrono::time_point<std::chrono::steady_clock> begin_t, last_refresh_t;
    Tp last_finished{0};
    double avg_df_per_it{0}, avg_t_per_it{0};
    const double alpha{0.99}; // EMA factor
    void print_time(uint64_t s) {
        std::cout << std::setfill('0') << std::setw(2) << s / 3600 << ":"
                << std::setfill('0') << std::setw(2) << (s % 3600) / 60 << ":"
                << std::setfill('0') << std::setw(2) << s % 60;
    }
    void print_bar(float prg) {
        static std::string boxes[] = {
            "\xE2\x96\x91", // ░
            "\xE2\x96\x8F", // ▏
            "\xE2\x96\x8E", // ▎
            "\xE2\x96\x8C", // ▌
            "\xE2\x96\x8D", // ▍
            "\xE2\x96\x88", // █
        };
        unsigned n_prg = prg * ncols * 5 + 0.5;
        const auto old_precision{std::cout.precision()};
        std::cout << std::setw(3) << std::setfill(' ') << std::fixed << std::setprecision(2) <<
                    prg * 100 << old_precision << "%|";
        for(unsigned i = 0; i < ncols; ++i) {
            if (5*i < n_prg)
                std::cout << boxes[n_prg - 5*i > 5 ? 5 : n_prg - 5*i];
            else
                std::cout << boxes[0];
        }
        std::cout << "|";
    }
public:
    ProgressBar(Tp total_, unsigned refresh_=10000, unsigned ncols_=20)
        : total(total_), refresh(refresh_), ncols(ncols_) {}
    void update(Tp finished, bool overwrite = true) {
        if (!inited || ++prt_cnt == refresh) {
            prt_cnt = 0;
            if (!inited || !overwrite)
                std::cout << "\n";
            if (overwrite)
                std::cout << "\r\033[0K";
            print_bar(float(finished)/total);
            std::cout << finished << "/" << total;

            auto cur_t = std::chrono::steady_clock::now();
            if (inited) {
                avg_df_per_it = avg_df_per_it * alpha + double(finished - last_finished)/refresh * (1-alpha);
                avg_t_per_it = avg_t_per_it * alpha +
                                std::chrono::duration_cast<std::chrono::seconds>(cur_t - last_refresh_t).count()
                                    / refresh * (1-alpha);

                std::cout << " ETA:";
                print_time(double(total - finished)/avg_df_per_it * avg_t_per_it);
                std::cout << ", E:";
                print_time(std::chrono::duration_cast<std::chrono::seconds>(cur_t - begin_t).count());
                std::cout << std::flush;
            } else {
                begin_t = cur_t;
                inited = true;
            }
            last_refresh_t = cur_t;
            last_finished = finished;
        }
    }
    void drop_total(Tp i) {
        assert(total >= i);
        total -= i;
    }
    double get_elapsed_time() const {
        return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - begin_t).count();
    }
    void end() {
        std::cout << "\n";
        inited = false;
    }
};


#endif // PROGRESSBAR_HH_
