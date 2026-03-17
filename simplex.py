import math

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

def inverse(A, identity):
    n = len(A)

    matrix_A = [line[:] for line in A]
    identity = [line[:] for line in identity]
    
    for i in range(n):
        pivotLine = i
        maxValue = math.fabs(matrix_A[i][i])

        for j in range(i + 1, n):
            if math.fabs(matrix_A[j][i]) > maxValue:
                pivotLine = j
                maxValue = math.fabs(matrix_A[j][i])

        if pivotLine != i:
            matrix_A[i], matrix_A[pivotLine] = matrix_A[pivotLine], matrix_A[i]
            identity[i], identity[pivotLine] = identity[pivotLine], identity[i]

        pivot = matrix_A[i][i]

        if pivot == 0:
            raise ValueError("Foi encontrada uma divisão por 0. Matriz singular.")

        for j in range(n):
            matrix_A[i][j] /= pivot
            identity[i][j] /= pivot

        for j in range(n):
            if i == j:
                continue

            multiplier = matrix_A[j][i]

            for k in range(n):
                matrix_A[j][k] -= multiplier * matrix_A[i][k]
                identity[j][k] -= multiplier * identity[i][k]

    return identity

if __name__ == "__main__":
    A = [[1, 2, 3], [3, 1, -1], [0, 4, 2]]
    identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    result = cofactor_expansion(A);
    print(result)

    if result != 0:
        inverse = inverse(A, identity)
        for line in inverse:
            print([round(element, 4) for element in line])
    else:
        print("Determinante é 0, logo não existe inversa")
