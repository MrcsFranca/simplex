def sub_matrix(A, line, column):
    sub = []
    for i in range(len(A)):
        if i == line:
            continue

        nl = []
        for j in range(len(A)):
            if j == column:
                continue

            nl.append(A[i][j])
        sub.append(nl)

    return sub

def cofactor_expansion(A):
    n = len(A)

    if n == 1:
        return A[0][0]

    det = 0
    for j in range(n):
        cofactor = ((-1) ** (0 + j)) * A[0][j]

        sub_mtx = sub_matrix(A, 0, j)

        det += cofactor * cofactor_expansion(sub_mtx)

    return det

if __name__ == "__main__":
    A = [[1, 2, 3], [3, 1, -1], [0, 4, 2]]
    result = cofactor_expansion(A);
    print(result)
