#include <utility>
#include <vector>
#include <iterator>
#include <algorithm>

template <class Itr>
struct enumerate {

  enumerate(Itr v) {
    auto first = std::begin(v);
    auto last = std::end(v);
    auto i = 0lu;
    auto make_pair = [&i] (auto item) {
      return std::make_pair(i++, item);
    };
    internal_container.resize(std::distance(first, last));
    std::transform(first, last, internal_container.begin(), make_pair);
  }

  auto begin() const {
    return internal_container.begin();
  }

  auto end() const {
    return internal_container.end();
  }

  std::vector<std::pair<size_t, typename Itr::value_type>> internal_container;
};
