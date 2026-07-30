#pragma once

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <variant>

struct Error {
    std::string message;
};

inline Error make_error(std::string msg) {
    return Error{std::move(msg)};
}

template<typename T>
class Result {
public:
    Result(T val) : data_(std::move(val)) {}
    Result(Error err) : data_(std::move(err)) {}

    bool ok() const { return data_.index() == 0; }
    explicit operator bool() const { return ok(); }

    T& operator*() { return std::get<0>(data_); }
    const T& operator*() const { return std::get<0>(data_); }
    T* operator->() { return &std::get<0>(data_); }
    const T* operator->() const { return &std::get<0>(data_); }

    const std::string& error() const { return std::get<1>(data_).message; }

    T take() {
        if (!ok()) {
            fprintf(stderr, "Result::take() called on error: %s\n",
                    std::get<1>(data_).message.c_str());
            abort();
        }
        return std::move(std::get<0>(data_));
    }

private:
    std::variant<T, Error> data_;
};

template<>
class Result<void> {
public:
    Result() = default;
    Result(Error err) : error_(std::move(err)) {}

    bool ok() const { return !error_.has_value(); }
    explicit operator bool() const { return ok(); }

    const std::string& error() const { return error_->message; }

    void take() {
        if (!ok()) {
            fprintf(stderr, "Result::take() called on error: %s\n",
                    error_->message.c_str());
            abort();
        }
    }

private:
    std::optional<Error> error_;
};
