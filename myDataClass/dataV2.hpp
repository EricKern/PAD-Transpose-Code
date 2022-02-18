#pragma once
#include <oneapi/tbb.h>
#include <memory>
#include <string.h>
#include <iomanip>


namespace pad {

template <typename T, typename A = std::allocator<T>>
class default_init_allocator : public A {
	// Implementation taken from https://stackoverflow.com/a/21028912.
   public:
	using A::A;

	template <typename U>
	struct rebind {
		using other = default_init_allocator<
		    U,
		    typename std::allocator_traits<A>::template rebind_alloc<U>>;
	};

	template <typename U>
	void construct(U* ptr) noexcept(
	    std::is_nothrow_default_constructible<U>::value) {
		::new (static_cast<void*>(ptr)) U;
	}
	template <typename U, typename... ArgsT>
	void construct(U* ptr, ArgsT&&... args) {
		std::allocator_traits<A>::construct(
		    static_cast<A&>(*this), ptr, std::forward<ArgsT>(args)...);
	}
};



template <typename T>
class arrayDataV2 {
    private:
        //Members
	    std::vector<T, default_init_allocator<T>> a;
        std::vector<T, default_init_allocator<T>> b;
        const size_t in_rows;               // nr. of rows of the input matrix (a)
        const size_t in_cols;               // nr. of columns of the input matrix (a)

        //private Methods
        void tbb_init(size_t gs, DataPartitioner part);
        void omp_init(size_t blocksize);

    public:
        arrayDataV2(const arrayDataV2&) = delete;
        arrayDataV2(arrayDataV2&&) = delete;
        arrayDataV2() = delete;
        // Serial Constructor
        explicit arrayDataV2(size_t in_rows,
                             size_t in_columns)
            : a(in_rows*in_columns), b(in_rows*in_columns), in_rows(in_rows), in_cols(in_columns){

            size_t num_elements = in_rows*in_columns;
            for (size_t i = 0; i < num_elements; ++i)
                a[i] = i;
            for (size_t i = 0; i < num_elements; ++i)
                b[i] = -1;
        } // end Serial Constructor

        // OMP Constructor
        explicit arrayDataV2(size_t in_rows,
                             size_t in_columns,
                             size_t blocksize,
                             const char* api)
            : a(in_rows*in_columns), b(in_rows*in_columns), in_rows(in_rows), in_cols(in_columns){
            if(strcmp(api, "OMP") == 0){
                omp_init(blocksize);
            }
            else{
                throw std::runtime_error("Wrong call of OMP Constructor in arrayDataV2 object");
            }
        } // end OMP Constructor

        // TBB Constructor
        explicit arrayDataV2(size_t in_rows,
                            size_t in_columns,
                            size_t blocksize,
                            const char* api,
                            DataPartitioner& part)
            : a(in_rows*in_columns), b(in_rows*in_columns), in_rows(in_rows), in_cols(in_columns){
            if(strcmp(api, "TBB") == 0){
                tbb_init(blocksize, part);
            }
            else{
                throw std::runtime_error("Wrong call of TBB Constructor in arrayDataV2 object");
            }
            
        } // end TBB Constructor

        auto get_range() {
            return std::make_tuple(std::make_tuple(a.begin(), a.end()),
                                std::make_tuple(b.begin(), b.end()));
        }
        auto get_ptr() {
            return std::make_tuple(a.data(), b.data());
        }

        void printA();
        void printB();
}; // end class arrayDataV2

template <typename T>
void arrayDataV2<T>::tbb_init(size_t gs, DataPartitioner part){

    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range2d<size_t>(0, in_rows, gs,
                                            0, in_cols, gs),
        [&](oneapi::tbb::blocked_range2d<size_t>& r) {
            for(size_t x = r.cols().begin(); x < r.cols().end(); ++x){
                for(size_t y = r.rows().begin(); y < r.rows().end(); ++y){
                    a[y * in_cols + x] = y * in_cols + x;
                    b[x * in_rows + y] = -1;
                }
            }
        },
        part);
}

template <typename T>
void arrayDataV2<T>::omp_init(size_t blocksize){
    // size_t out_rows = in_cols;
	size_t out_columns = in_rows;

    #pragma omp parallel for schedule(DATA_POLICY)
    for (size_t xx = blocksize; xx <= in_cols; xx+=blocksize){
        for(size_t yy = blocksize; yy <= in_rows; yy+=blocksize){
            for (size_t x = xx - blocksize; x < xx; ++x){
                for (size_t y = yy - blocksize; y < yy; ++y){
                    a[x + y * in_cols] = x + y * in_cols;
                    b[y + x * out_columns] = -1;
                }
            }
        }
    }

    size_t restCols = in_cols % blocksize;
    size_t restRows = in_rows % blocksize;

    size_t restStartC = in_cols - restCols;
    size_t restStartR = in_rows - restRows;

    // calculate rest Bottom to restRight (relative to Input Matrix)
    #pragma omp parallel for schedule(DATA_POLICY)
    for (size_t x = 0; x < restStartC; ++x) {
        for (size_t y = restStartR; y < in_rows; ++y) {
            a[x + y * in_cols] = x + y * in_cols;
            b[y + x * out_columns] = -1;
        }
    }
	// calculate rest Right to Bottom (relative to Input Matrix)
    #pragma omp parallel for schedule(DATA_POLICY)
    for (size_t x = restStartC; x < in_cols; ++x) {
        for (size_t y = 0; y < in_rows; ++y) {
            a[x + y * in_cols] = x + y * in_cols;
            b[y + x * out_columns] = -1;
        }
    }
}

template <typename T>
void arrayDataV2<T>::printA(){
    for (size_t y = 0; y < in_rows; y++)
    {
        for (size_t x = 0; x < in_cols; x++)
        {
            std::cout << std::setw(4) << a[x + y*in_cols] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
void arrayDataV2<T>::printB(){
    size_t out_rows = in_cols;
    size_t out_cols = in_rows;
    for (size_t y = 0; y < out_rows; y++)
    {
        for (size_t x = 0; x < out_cols; x++)
        {
            std::cout << std::setw(4) << b[x + y*out_cols] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}



} // namespace pad

