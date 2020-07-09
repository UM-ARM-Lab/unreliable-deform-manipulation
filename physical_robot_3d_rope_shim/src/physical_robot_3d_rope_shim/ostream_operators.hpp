#ifndef LBV_OSTREAM_OPERATORS
#define LBV_OSTREAM_OPERATORS

#include <ostream>
#include <typeinfo>
#include <moveit/collision_detection/collision_common.h>

static std::ostream& operator<<(std::ostream& out, collision_detection::CollisionResult const& cr)
{
  out << "  collision:      " << cr.collision << "\n"
      << "  distance:       " << cr.distance << "\n"
      << "  contact_count:  " << cr.contact_count << "\n"
      << "  contacts:\n";
  for (auto const& [names, contact_list] : cr.contacts)
  {
    out << "    " << names.first << "," << names.second << "\n";
    for (auto const& contact : contact_list)
    {
      out << "      pos:          " << contact.pos.transpose() << "\n"
          << "      normal:       " << contact.normal.transpose() << "\n"
          << "      depth:        " << contact.depth << "\n"
          << "      body_type_1:  " << contact.body_type_1 << " name: " << contact.body_name_1 << "\n"
          << "      body_type_2:  " << contact.body_type_2 << " name: " << contact.body_name_2 << "\n";
    }
  }
  out << std::flush;
  return out;
}

template <typename T>
static std::ostream& operator<<(std::ostream& out, std::vector<T> const& vec)
{
  for (auto const& val: vec)
  {
    out << val;
    if (typeid(T) != typeid(std::string))
    {
      out << " ";
    }
    else
    {
      out << "\n";
    }
    
     
  }
  return out;
}

#endif
