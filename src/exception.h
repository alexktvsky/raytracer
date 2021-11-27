#ifndef EXCEPTION_H
#define EXCEPTION_H

#include <string>
#include <exception>

#define ExceptionFromHere(message)  Exception(__FILE__, __LINE__, message)


class Exception : public std::exception {
public:
    Exception(const std::string &file, int line, const std::string &message);
    const char *what(void) const noexcept;
    std::string getMessage(void) const;
    std::string getFile(void) const;
    int getLine(void) const;
private:
    std::string m_file;
    int m_line;
    std::string m_message;
    std::string m_exp_string;
}; // End of class


inline const char *Exception::what(void) const noexcept
{
    return m_exp_string.c_str();
}

inline std::string Exception::getMessage(void) const
{
    return m_message;
}

inline std::string Exception::getFile(void) const
{
    return m_file;
}

inline int Exception::getLine(void) const
{
    return m_line;
}

#endif // EXCEPTION_H
