#ifndef SENTENCE_ENDED_HH_
#define SENTENCE_ENDED_HH_

#include <string>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <codecvt>
#include <locale>
#include <cassert>
#include <memory>
#include <iostream>
#include <algorithm>

#if __cplusplus < 201402L
// Define make_unique for pre-C++14 compatibility
namespace std{
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
}
#endif

class TrieNode {
public:
    bool isEndOfWord;
    std::unordered_map<char, std::unique_ptr<TrieNode>> children;

    TrieNode() : isEndOfWord(false) {}
};

class Trie {
public:
    Trie() : root(std::make_unique<TrieNode>()) {}

    void insert(const std::string& word) {
        TrieNode* node = root.get();
        for (auto it = word.rbegin(); it != word.rend(); ++it) {
            if (!node->children[*it]) {
                node->children[*it] = std::make_unique<TrieNode>();
            }
            node = node->children[*it].get();
        }
        node->isEndOfWord = true;
    }

    bool endsWith(const std::string& word) const {
        TrieNode* node = root.get();
        for (auto it = word.rbegin(); it != word.rend(); ++it) {
            if (node->isEndOfWord) {
                return true;
            }
            if (!node->children.count(*it)) {
                return false;
            }
            node = node->children.at(*it).get();
        }
        return node->isEndOfWord;
    }

private:
    std::unique_ptr<TrieNode> root;
};

static std::wstring s_to_wstring(const std::string &str) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.from_bytes(str);
}
static std::string w_to_string(const std::wstring &wstr) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.to_bytes(wstr);
}

static bool sentence_ended(const std::string &inp) {
    static bool inited;
    static Trie trie;
    if (!inited) {
        std::unordered_set<unsigned> terminators_unicode = {
            0x002E, // .
            0x3002, // 。
            0xFF0E, // ．
            0xFF61, // ｡
            0x003F, // ?
            0xFF1F, // ？ 
            0x0021, // !
            0xFF01 // ！
        };
        std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> convert;
        for(auto uni : terminators_unicode) {
            std::string utf8 = convert.to_bytes(uni);
            trie.insert(utf8);
        }
        inited = true;
    }
    std::wstring trim = s_to_wstring(inp);
    auto it = std::find_if_not(trim.rbegin(), trim.rend(), std::iswspace).base();
    trim.erase(it, trim.end());

    return trie.endsWith(w_to_string(trim));
}

#ifdef TESTUNIT

void test_insert_and_search() {
    Trie trie;
    trie.insert("hello");
    trie.insert("world");
    
    assert(trie.endsWith("hello") == true);
    assert(trie.endsWith("world") == true);
    assert(trie.endsWith("hell") == false);
    assert(trie.endsWith("orl") == false);
    assert(trie.endsWith("worldly") == false);
    std::cout << "test_insert_and_search passed." << std::endl;
}

void test_sentence_ended() {
    assert(sentence_ended("The sentence ends here.") == true);
    assert(sentence_ended("这是一个句子。") == true);
    assert(sentence_ended("This does not") == false);
    assert(sentence_ended("What about this?") == true);
    assert(sentence_ended("Oh!") == true);
    assert(sentence_ended("") == false);  // Empty string
    assert(sentence_ended("No punctuation") == false);
    assert(sentence_ended("Multiple terminators!..") == true);
    std::cout << "test_sentence_ended passed." << std::endl;
}

void test_unicode_handling() {
    assert(sentence_ended("こんにちは。") == true);
    assert(sentence_ended("こんにちは") == false);
    assert(sentence_ended("¿Cómo estás?") == true);
    std::cout << "test_unicode_handling passed." << std::endl;
}

int main(int argc, char *argv[]) {
    test_insert_and_search();
    test_sentence_ended();
    test_unicode_handling();
    std::cout << "All tests passed!" << std::endl;

    printf("%d\n", sentence_ended(argv[1]));
    return 0;
}

#endif

#endif 