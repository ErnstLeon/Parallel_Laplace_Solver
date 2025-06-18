#ifndef MATRIX_TYPE_H
#define MATRIX_TYPE_H

#include <cstdlib>
#include <vector>

namespace flow
{
template<typename T>
class matrix {
private:
    std::vector<T> values{};
    int x_dim{};
    int y_dim{};

public: 

    matrix() = default;

    matrix(const matrix<T> & matrix_orig) : 
        values(matrix_orig.values), x_dim(matrix_orig.x_dim), y_dim(matrix_orig.y_dim){}
    
    matrix(matrix<T> && matrix_orig) noexcept : 
        values(std::move(matrix_orig.values)), x_dim(matrix_orig.x_dim), y_dim(matrix_orig.y_dim) {}

    matrix(int x_dim, int y_dim) : 
        values(x_dim * y_dim, static_cast<T>(0)), x_dim(x_dim), y_dim(y_dim) {}

    matrix(int x_dim, int y_dim, T init) : 
        values(x_dim * y_dim, init), x_dim(x_dim), y_dim(y_dim) {}

    matrix<T> & operator = (const matrix<T> & matrix_orig){
        if(this != &matrix_orig){
            values = matrix_orig.values;
            x_dim = matrix_orig.x_dim;
            y_dim = matrix_orig.y_dim;
        }
        return *this;
    }

    matrix<T> & operator = (matrix<T> && matrix_orig) noexcept {
     
        if (this != &matrix_orig) {
            values = std::move(matrix_orig.values);
            x_dim = matrix_orig.x_dim;
            y_dim = matrix_orig.y_dim;

            matrix_orig.values.clear();
            matrix_orig.x_dim = static_cast<int>(0);
            matrix_orig.y_dim = static_cast<int>(0);
        }

        return *this;
    }

    T& operator() (int i, int j){
        return values[y_dim * i + j];
    }

    const T& operator() (int i, int j) const {
        return values[y_dim * i + j];
    }

    T& at(int i, int j){
        if (i >= x_dim || j >= y_dim) {
            throw std::out_of_range("Matrix index out of bounds!");
        }
        return values.at(y_dim * i + j);
    }

    const T& at(int i, int j) const {
        if (i >= x_dim || j >= y_dim) {
            throw std::out_of_range("Matrix index out of bounds!");
        }
        return values.at(y_dim * i + j);
    }

    constexpr int size() const {
        return x_dim * y_dim;
    }

    constexpr int size_x() const {
        return x_dim;
    }

    constexpr int size_y() const {
        return y_dim;
    }

    void fill(T fill_num) {
        std::fill(values.begin(), values.end(), fill_num);
    }

    void resize(int new_x_dim, int new_y_dim) {
        x_dim = new_x_dim;
        y_dim = new_y_dim;
        values.resize(x_dim * y_dim);
    }

    auto data() {
        return values.data();
    }

    auto data(int i, int j) {
        if (i >= x_dim || j >= y_dim) {
            throw std::out_of_range("Matrix index out of bounds!");
        }
        return values.data() + (y_dim * i + j);
    }

    void clear() {
        values.clear();
        x_dim = static_cast<int>(0);
        y_dim = static_cast<int>(0);
    }

    matrix<T> get_submatrix(int start_x, int size_x, int start_y, int size_y) const {
        
        if (start_x + size_x > this->x_dim || start_y + size_y > this->y_dim) {
            throw std::out_of_range("Matrix index out of bounds!");
        }
        
        matrix<T> tmp_matrix (size_x, size_y, static_cast<T>(0));

        for(int i = 0; i < size_x; ++i){
            for(int j = 0; j < size_y; ++j){
                tmp_matrix(i,j) = (*this)(start_x + i, start_y + j);
            }
        }
        return tmp_matrix;
    }

    void reshape_to_submatrix(int start_x, int size_x, int start_y, int size_y) {
        
        if (start_x + size_x > this->x_dim || start_y + size_y > this->y_dim) {
            throw std::out_of_range("Matrix index out of bounds!");
        }
        
        std::vector<T> new_values(size_x * size_y);

        for(int i = 0; i < size_x; ++i){
            for(int j = 0; j < size_y; ++j){
                new_values[size_y * i + j] = (*this)(start_x + i, start_y + j);
            }
        }

        values = std::move(new_values);
        x_dim = size_x;
        y_dim = size_y;

    }

};
}

#endif // MATRIX_TYPE_H