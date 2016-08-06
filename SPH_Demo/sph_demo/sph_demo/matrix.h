#ifndef MATRIX_H
#define MATRIX_H

#include <ostream>
#include <cassert>

template<int M, int N>
class matrix
{
public:
    matrix();

    void set(int i, int j, float a);
    float get(int i, int j) const;
    void zero();

    void add(
        const matrix<M,N>& A, const matrix<M, N>& B
        );
    void add(
        const matrix<M,N>& B
        );
    void sub(
        const matrix<M,N>& A, const matrix<M, N>& B
        );
    void sub(
        const matrix<M,N>& B
        );
    template<int Z>
    void mult(
        const matrix<M,Z>& A, const matrix<Z, N>& B
        );
    void premult(
        const matrix<M,M>& A
        );
    void postmult(
        const matrix<N,N>& A
        );
    
    float& operator()(int i, int j);
    const float& operator()(int i, int j) const;
    
    matrix<M,N> operator+(const matrix<M,N>& B) const;
    matrix<M,N> operator-(const matrix<M,N>& B) const;
    void operator+=(const matrix<M,N>& B);
    void operator-=(const matrix<M,N>& B);
    matrix<M,N> operator*(float s) const;
    matrix<M,N> operator*(const matrix<M,N>& B) const;


private:
    float val[M*N];
};

typedef matrix<2,2> matrix22;
typedef matrix<2,3> matrix23;
typedef matrix<3,2> matrix32;
typedef matrix<4,3> matrix43;
typedef matrix<3,4> matrix34;
typedef matrix<4,4> matrix44;
    
template<int M, int N>
matrix<M,N>::matrix()
{
    zero();
}
    
template<int M, int N>
void matrix<M,N>::add(
    const matrix<M,N>& A, const matrix<M, N>& B
    )
{
    // ith row
    for (int i = 0; i < M; ++i) {
        // jth column
        for (int j = 0; j < N; ++j) {
            set(i, j, A(i,j)+B(i,j));
        }
    }
}

template<int M, int N>
void matrix<M,N>::add(
    const matrix<M,N>& B
    )
{
    for (int i = 0; i < M; ++i) {
        // jth column
        for (int j = 0; j < N; ++j) {
            set(i, j, get(i,j)+B(i,j));
        }
    }
}

template<int M, int N>
void matrix<M,N>::sub(
    const matrix<M,N>& A, const matrix<M, N>& B
    )
{
    for (int i = 0; i < M; ++i) {
        // jth column
        for (int j = 0; j < N; ++j) {
            set(i, j, A(i,j)-B(i,j));
        }
    }
}

template<int M, int N>
void matrix<M,N>::sub(
    const matrix<M,N>& B
    )
{
    for (int i = 0; i < M; ++i) {
        // jth column
        for (int j = 0; j < N; ++j) {
            set(i, j, get(i,j)-B(i,j));
        }
    }
}
    
template<int M, int N>
template<int Z>
void matrix<M,N>::mult(
    const matrix<M,Z>& A, const matrix<Z, N>& B
    )
{
    // ith row
    for (int i = 0; i < M; ++i) {
        // jth column
        for (int j = 0; j < N; ++j) {
            float val = 0;
            for (int k = 0; k < Z; ++k) {
                val += A(i,k) * B(k, j);
            }
            set(i,j, val);
        }
    }
}
    
template<int M, int N>
void matrix<M,N>::premult(
    const matrix<M,M>& A
    )
{
    matrix<M,N> tmp(*this);
    mult(A, tmp);
}

template<int M, int N>
void matrix<M,N>::postmult(
    const matrix<N,N>& A
    )
{
    matrix<M,N> tmp(*this);
    mult(tmp,A);
}
    
template<int M, int N>
void matrix<M,N>::set(int i, int j, float a)
{
    val[i+M*j] = a;
}
    
template<int M, int N>
float matrix<M,N>::get(int i, int j) const
{
    return val[i+M*j];
}
    
template<int M, int N>
void matrix<M,N>::zero()
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            set(i, j, 0);
        }
    }
}

template<int M, int N>
float& matrix<M,N>::operator()(int i, int j)
{
    return val[i+M*j];
}

template<int M, int N>
const float& matrix<M,N>::operator()(int i, int j) const
{
    return val[i+M*j];
}

template<int M, int N>
matrix<M,N> matrix<M,N>::operator+(const matrix<M,N>& B) const 
{
    matrix<M,N> AB(*this);
    AB.add(B);
    return AB;
}

template<int M, int N>
matrix<M,N> matrix<M,N>::operator-(const matrix<M,N>& B) const 
{
    matrix<M,N> AB(*this);
    AB.sub(B);
    return AB;
}

template<int M, int N>
void matrix<M,N>::operator+=(const matrix<M,N>& B) 
{
    add(B);
}

template<int M, int N>
void matrix<M,N>::operator-=(const matrix<M,N>& B)
{
    sub(B);
}

template<int M, int N>
matrix<M,N> matrix<M,N>::operator*(float s) const 
{
    matrix<M,N> B(*this);
    B.scale(s);
    return B;
}
    
template<int M, int N>
matrix<M,N> matrix<M,N>::operator*(const matrix<M,N>& B) const
{
    matrix<M,N> AB;
    AB.mult(*this, B);
    return AB;
}
    
template<int M, int N>
std::ostream& operator<<(std::ostream& ostr, const matrix<M,N>& A)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            ostr << A.get(i, j) << " ";
        }
        ostr << std::endl;
    }
    return ostr;
}


#endif
