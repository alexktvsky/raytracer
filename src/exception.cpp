#include "exception.h"


Exception::Exception(const std::string &file, int line, const std::string &message)
    : m_file(file)
    , m_line(line)
    , m_message(message)
    , m_exp_string(m_file + ":" + std::to_string(m_line) + ": " + m_message)
{}
