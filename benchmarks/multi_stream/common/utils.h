#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <regex>
#include "dvrapi_error_string.h"

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(int err, const char *file, const int line) {
  if (0 != err) {
    fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, getCudaDrvErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

template<typename T> struct map_init_helper
{
    T& data;
    map_init_helper(T& d) : data(d) {}
    map_init_helper& operator() (typename T::key_type const& key, typename T::mapped_type const& value)
    {
        data[key] = value;
        return *this;
    }
};

template<typename T> map_init_helper<T> map_init(T& item)
{
    return map_init_helper<T>(item);
}

template<typename I, typename T>
inline bool compare(const std::tuple<I, I, T, I> &lhs, const std::tuple<I, I, T, I> &rhs) {
	I a = std::get < 0 > (lhs);
	I b = std::get < 0 > (rhs);
	I c = std::get < 1 > (lhs);
	I d = std::get < 1 > (rhs);
	if (a == b)
		return c < d;
	else
		return a < b;
}

template<typename I>
inline bool compare(const std::tuple<I, I, I> &lhs, const std::tuple<I, I, I> &rhs) {
	I a = std::get < 0 > (lhs);
	I b = std::get < 0 > (rhs);
	I c = std::get < 1 > (lhs);
	I d = std::get < 1 > (rhs);
	if (a == b)
		return c < d;
	else
		return a < b;
}


template<typename I, typename T>
inline void customSort(I *row_indices, I *col_indices, T *values, I nnz) {
	I nvals = nnz;
	std::vector<std::tuple<I, I, T, I>> my_tuple;

	for (I i = 0; i < nvals; ++i)
		my_tuple.push_back(std::make_tuple(row_indices[i], col_indices[i], values[i], i));

	std::sort(my_tuple.begin(), my_tuple.end(), compare<I, T>);

	for (I i = 0; i < nvals; ++i) {
		row_indices[i] = std::get<0>(my_tuple[i]);
		col_indices[i] = std::get<1>(my_tuple[i]);
		values[i] = std::get<2>(my_tuple[i]);
	}
}

template<typename I, typename T>
inline void coo2csr(I *csrRowPtr, I *csrColInd, T *csrVal, I *row_indices,
		I* col_indices, T* values, size_t nrows, size_t ncols, size_t nnz) {

	I temp, row, col, dest, cumsum = 0;

	std::vector<I> row_indices_t(row_indices, row_indices + nnz);
	std::vector<I> col_indices_t (col_indices, col_indices + nnz);
	std::vector<T> values_t(values, values + nnz);

	customSort<I, T>(row_indices_t.data(), col_indices_t.data(), values_t.data(), nnz);

	// Set all rowPtr to 0
	for (I i = 0; i <= nrows; i++)
		csrRowPtr[i] = 0;

	// Go through all elements to see how many fall in each row
	for (I i = 0; i < nnz; i++) {
		row = row_indices_t[i];
		if (row >= nrows)
			std::cout << "Error: Index out of bounds!\n";
		csrRowPtr[row]++;
	}

	// Cumulative sum to obtain rowPtr
	for (I i = 0; i < nrows; i++) {
		temp = csrRowPtr[i];
		csrRowPtr[i] = cumsum;
		cumsum += temp;
	}
	csrRowPtr[nrows] = nnz;

	// Store colInd and val
	for (I i = 0; i < nnz; i++) {
		row = row_indices_t[i];
		dest = csrRowPtr[row];
		col = col_indices_t[i];
		if (col >= ncols)
			std::cout << "Error: Index out of bounds!\n";
		csrColInd[dest] = col;
		csrVal[dest] = values_t[i];
		csrRowPtr[row]++;
	}
	cumsum = 0;

	// Undo damage done to rowPtr
	for (I i = 0; i < nrows; i++) {
		temp = csrRowPtr[i];
		csrRowPtr[i] = cumsum;
		cumsum = temp;
	}
	temp = csrRowPtr[nrows];
	csrRowPtr[nrows] = cumsum;
	cumsum = temp;
}

template<typename T>
void write_to_file(T * data, size_t size, std::string file_path)
{
	std::fstream fs(file_path,std::ios::out);
	if(fs.is_open())
	{
		for(size_t i = 0; i < size; i++)
		{
			fs<<data[i]<<" ";
		}
		fs.close();
	}
	else
	{
		std::cout<<"Cannot open the file of "<<file_path<<std::endl;
		exit(1);
	}
}

template<typename T>
void read_file(T * data, size_t size, std::string file_path)
{
	std::fstream fs(file_path,std::ios::in);
	if(fs.is_open())
	{
		for(size_t i = 0; i < size; i++)
		{
			T num;
			fs>>num;
			data[i] = num;
		}
		fs.close();
	}
	else
	{
		std::cout<<"Cannot open the file of "<<file_path<<std::endl;
		exit(1);
	}
}
