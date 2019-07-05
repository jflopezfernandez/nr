
#include "nr.h"

#define SWAP(a, b)  {                   \
                        temp = (a);     \
                        (a)  = (b);     \
                        (b)  = temp;    \
                    }

#define NR_END 1
#define FREE_ARG char*

/* Numerical Recipes standard error handler */
void nrerror(const char* errmsg);

/* Allocate an int vector with subscript range v[nl..nh] */
int* ivector(long nl, long nh);

/* Free an int vector allocated with ivector() */
void free_ivector(int* v, long nl, long nh);

/* Allocate a float matrix with subscript range m[nrl..nrh][ncl..nch] */
float** matrix(long nrl, long nrh, long ncl, long nch);

/* Free a float matrix allocated by matrix() */
void free_matrix(float** m, long nrl, long nrh, long ncl, long nch);

void gaussj(float** a, int n, float** b, int m);

__attribute__((noreturn))
void nrerror(const char* errmsg) {
    fprintf(stderr, "[ERROR] %s\n", errmsg);
    exit(EXIT_FAILURE);
}

int* ivector(long nl, long nh) {
    int* v;

    v = (int *) malloc((size_t) ((nh - nl + 1 + NR_END) * sizeof (int)));

    if (!v) {
        nrerror("Allocation failure in ivector()");
    }

    return v - nl + NR_END;
}

void free_ivector(int* v, long nl, long nh) {
    free((FREE_ARG) (v + nl - NR_END));
}

float** matrix(long nrl, long nrh, long ncl, long nch) {
    long i, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
    float** m;

    /* Allocate pointers to rows */
    m = (float **) malloc((size_t) ((nrow + NR_END) * sizeof(float *)));

    if (!m) {
        nrerror("Allocation failure 1 in matrix()");
    }

    m += NR_END;
    m -= nrl;

    /* Allocate rows and set pointers to them */
    m[nrl] = (float *) malloc((size_t) ((nrow * ncol + NR_END) * sizeof (float)));

    if (!m[nrl]) {
        nrerror("Allocation failure 2 in matrix()");
    }

    m[nrl] += NR_END;
    m[nrl] -= ncl;

    for (i = nrl + 1; i <= nrh; i++) {
        m[i] = m[i - 1] + ncol;
    }

    /* Return pointer to array of pointers to rows */
    return m;
}

void free_matrix(float** m, long nrl, long nrh, long ncl, long nch) {
    free((FREE_ARG) (m[nrl] + ncl - NR_END));
    free((FREE_ARG) (m + nrl - NR_END));
}

void gaussj(float** a, int n, float** b, int m) {
    int *indxc, *indxr, *ipiv;
    int i, icol, irow, j, k, l, ll;
    float big, dum, pivinv, temp;

    indxc = ivector(1, n);
    indxr = ivector(1, n);
    ipiv  = ivector(1, n);

    for (j = 1; j <= n; j++) {
        ipiv[j] = 0;
    }

    for (i = 1; i <= n; i++) {
        big = 0.0;

        for (j = 1; j <= n; j++) {
            if (ipiv[j] != 1) {
                for (k = 1; k <= n; k++) {
                    if (ipiv[k] == 0) {
                        if (fabs(a[j][k]) >= big) {
                            big = fabs(a[j][k]);

                            irow = j;
                            icol = k;
                        }
                    } else if (ipiv[k] > 1) {
                        nrerror("gaussj: Singular Matrix - 1");
                    }
                }
            }
        }

        ++(ipiv[icol]);

        /** We now have the pivot element, so we interchange rows, if needed,
         *  to put the pivot element on the diagonal. The columns are not 
         *  physically interchanged, only relabeled: indxc[i], the column of the
         *  ith pivot element, is the ith column that is reduced, while indxr[i]
         *  is the row in which that pivot element was originally located. If
         *  indxr[i] != indxc[i] there is an implied column interchange. With
         *  this form of bookkeeping, the solution b's will end up in the 
         *  correct order, and the inverse matrix will be scrambled by columns.
         * 
         */
        if (irow != icol) {
            for (l = 1; l <= n; l++) {
                SWAP(a[irow][l], a[icol][l]);
            }

            for (l = 1; l <= m; l++) {
                SWAP(b[irow][l], b[icol][l]);
            }
        }

        /** We are now ready to divide the pivot row by the pivot element,
         *  located at the irow and icol.
         * 
         */
        indxr[i] = irow;
        indxc[i] = icol;

        if (a[icol][icol] == 0.0) {
            nrerror("gaussj: Singular Matrix - 2");
        }

        pivinv = 1.0 / a[icol][icol];
        a[icol][icol] = 1.0;

        for (l = 1; l <= n; l++) {
            a[icol][l] *= pivinv;
        }

        for (l = 1; l <= m; l++) {
            b[icol][l] *= pivinv;
        }

        for (ll = 1; ll <= n; ll++) {
            if (ll != icol) {
                dum = a[ll][icol];
                a[ll][icol] = 0.0;

                for (l = 1; l <= n; l++) {
                    a[ll][l] -= a[icol][l] * dum;
                }

                for (l = 1; l <= m; l++) {
                    b[ll][l] -= b[icol][l] * dum;
                }
            }
        }
    }

    /** This is the end of the main loop over columns of the reduction. It only
     *  remains to unscramble the solution in view of the column interchanges.
     *  We do this by interchanging pairs of columns in the reverse order that 
     *  the permutation was built up.
     * 
     */
    for (l = n; l >= 1; l--) {
        if (indxr[l] != indxc[l]) {
            for (k = 1; k <= n; k++) {
                SWAP(a[k][indxr[l]], a[k][indxc[l]]);
            }
        }
    }

    free_ivector(ipiv, 1, n);
    free_ivector(indxr, 1, n);
    free_ivector(indxc, 1, n);
}

#define NP 20
#define MP 20
#define MAXSTR 80

int main(int argc __attribute__((unused)), char *argv[] __attribute__((unused)))
{
    int j, k, l, m, n;
    float**a, **ai, **u, **b, **x, **t;
    char dummy[MAXSTR];
    FILE* fp;

    a = matrix(1, NP, 1, NP);
    ai = matrix(1, NP, 1, NP);
    u = matrix(1, NP, 1, NP);
    b = matrix(1, NP, 1, MP);
    x = matrix(1, NP, 1, MP);
    t = matrix(1, NP, 1, MP);

    if ((fp = fopen("matrix1.dat", "r")) == NULL) {
        nrerror("Data file matrix1.dat not found\n");
    }

    while (!feof(fp)) {
        fgets(dummy, MAXSTR, fp);
        fgets(dummy, MAXSTR, fp);
        fscanf(fp, "%d %d ", &n, &m);
        fgets(dummy, MAXSTR, fp);

        printf("\n%d x %d Matrix with 1 x %d solution vector:\n\n", n, n, m);

        for (k = 1; k <= n; k++) {
            for (l = 1; l <= n; l++) {
                fscanf(fp, "%f ", &a[k][l]);
            }
        }

        /* Print matrix */
        for (size_t row = 1; row <= (size_t) n; ++row) {
            for (size_t column = 1; column <= (size_t) n; ++column) {
                printf("%4.2f ", a[row][column]);
            }

            printf("\n");
        }

        fgets(dummy, MAXSTR, fp);

        for (l = 1; l <= m; l++) {
            for (k = 1; k <= n; k++) {
                fscanf(fp, "%f ", &b[k][l]);
            }
        }

        /* Save matrices for later testing of results */
        for (l = 1; l <= n; l++) {
            for (k = 1; k <= n; k++) {
                ai[k][l] = a[k][l];
            }
            
            for (k = 1; k <= m; k++) {
                x[l][k] = b[l][k];
            }
        }

        /* Invert matrix */
        gaussj(ai, n, x, m);

        printf("\nInverse of matrix: \n");
        
        for (k = 1; k <= n; k++) {
            for (l = 1; l <= n; l++) {
                printf("%12.6f", ai[k][l]);
            }

            printf("\n");
        }

        /* Check inverse */
        printf("\nA * A': \n\n");

        for (k = 1; k <= n; k++) {
            for (l = 1; l <= n; l++) {
                u[k][l] = 0.0;

                for (j = 1; j <= n; j++) {
                    u[k][l] += (a[k][j] * ai[j][l]);
                }
            }

            for (l = 1; l <= n; l++) {
                printf("%12.6f", u[k][l]);
            }

            printf("\n");
        }

        /* Check vector solutions */
        printf("\nCheck the following for equality: \n");
        printf("%21s %14s\n", "original", "matrix * sol'n\n");

        for (l = 1; l <= m; l++) {
            printf("vector %2d: \n", l);

            for (k = 1; k <= n; k++) {
                t[k][l] = 0.0;

                for (j = 1; j<= n; j++) {
                    t[k][l] += (a[k][j] * x[j][l]);
                }

                printf("%8s %12.6f %12.6f\n", " ", b[k][l], t[k][l]);
            }
        }

        printf("*********************************** \n");
        printf("press RETURN for next problem: \n");
        getchar();
    }

    fclose(fp);

    free_matrix(t, 1, NP, 1, MP);
    free_matrix(x, 1, NP, 1, MP);
    free_matrix(b, 1, NP, 1, MP);
    free_matrix(u, 1, NP, 1, NP);
    free_matrix(ai, 1, NP, 1, NP);
    free_matrix(a, 1, NP, 1, NP);

    return EXIT_SUCCESS;
}

