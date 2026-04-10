import math
import re
import numpy as np
import random

random.seed()

#Verifica se o coeficiente é positivo ou negativo
def extract_coefficient(coefficient):
    coefficient = coefficient.replace(" ", "")
    if coefficient == "" or coefficient == "+":
        return 1.0
    if coefficient == "-":
        return -1.0
    return float(coefficient)

#Faz a leitura do txt e algumas padronizações
def read_txt(file_path):
    pattern = re.compile(r'([+-]?\s*\d*(?:\.\d+)?)\s*[xX](\d+)')

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    empty_lines = []
    max_index = 0

    for line in lines:
        line_normalized = line.replace('−', '-').replace('≤', '<=').replace('≥', '>=').strip()
        if not line_normalized:
            continue
            
        if re.search(r'>=\s*0', line_normalized) and ',' in line_normalized:
            continue

        matches = pattern.findall(line_normalized)
        for _, idx_str in matches:
            max_index = max(max_index, int(idx_str))

        empty_lines.append(line_normalized)

    n_vars = max_index

    func_type = "max"
    vector_c = np.zeros(n_vars, dtype=float)
    matrix_A_list = []
    vector_b_list = []
    sinals = []

    line_obj = empty_lines[0]
    if "min" in line_obj.lower():
        func_type = "min"
        
    if "=" in line_obj:
        equation = line_obj.split("=")[1]
        for coef_str, idx_str in pattern.findall(equation):
            vector_c[int(idx_str) - 1] = extract_coefficient(coef_str)

    for line in empty_lines[1:]:
        sinal = None
        if '<=' in line: sinal = '<='
        elif '>=' in line: sinal = '>='
        elif '=' in line: sinal = '='
        else: continue

        left, right = line.split(sinal)
        
        line_A = np.zeros(n_vars, dtype=float)
        for coef_str, idx_str in pattern.findall(left):
            line_A[int(idx_str) - 1] = extract_coefficient(coef_str)
            
        matrix_A_list.append(line_A)
        vector_b_list.append(float(right.strip()))
        sinals.append(sinal)

    matrix_A = np.array(matrix_A_list, dtype=float)
    vector_b = np.array(vector_b_list, dtype=float)

    return func_type, vector_c, matrix_A, vector_b, sinals

#faz a multiplicação das matrizes
def multiply(A, B):
    rows_A = len(A)
    cols_A = len(A[0]) if rows_A > 0 else 0
    
    rows_B = len(B)
    cols_B = len(B[0]) if rows_B > 0 else 0

    if cols_A != rows_B:
        raise ValueError(f"colunas da matriz A precisa ser igual número de linhas da matriz B")

    result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]

    return result

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

#normaliza as restrições - forma padrão
def normalized_func(A, sinals, c):
    # se a inegualdade for menor eu somo uma variavel
    # se a inegualdade for maior eu subtraio uma variavel
    A = np.array(A, dtype=float)
    c = np.array(c, dtype=float)
    n = A.shape[0]
    n_var = sum(1 for sinal in sinals if sinal in ['<=', '>='])
    if n_var == 0:
        return A

    matrix_vars = np.zeros((n, n_var))

    idx = 0
    for i in range(n):
        lines = [0] * n_var
        if sinals[i] == '>=':
            matrix_vars[i, idx] = -1
            idx += 1
            c = np.hstack((c, 0))
        elif sinals[i] == '<=':
            matrix_vars[i, idx] = 1
            idx += 1
            c = np.hstack((c, 0))

    A = np.hstack((A, matrix_vars))

    return A, c
"""
    n = len(A)
    k = len(A[0])

    aux = [[0.0] for _ in range(n)]
    print(f'vetor com quantidade de colunas: {aux}')

    n_var = 0
    for sinal in sinals:
        if sinal == '<=' or sinal == '>=':
            n_var += 1

    if n_var == 0:
        return A

    for i in range(n_var):
        for j in range(1, k):
            A = np.append(A, aux, axis=1)

    print(A)


    print(f'matriz depois de estar normalizada:\n {A}')
    return A
"""

# Se a função for max, mult por -1 para trabalhar só com mínimo
def all_min(c):
    n = len(c)
    for i in range(n):
        c[i] = c[i] * -1

    return c

#Define a matriz básica e a matriz não básica
def basic_nonBasic(A, b):
    n_lines = len(A)
    n_columns = len(A[0]) if n_lines > 0 else 0
    B = np.zeros((n_lines, n_lines), dtype=float)
    NB = np.zeros((n_lines, n_lines), dtype=float)
    sorted_values = []
    print(f"tamanho de A: {n_lines}")
    sorted_values = random.sample(range(n_columns), n_lines)
    print(f"valores sorteados {sorted_values}")
    for i in range(n_lines):
        pos = sorted_values[i]
        print(f"valor de pos: {pos}")
        print("coluna de a")
        print(A[1])
        #print(A[pos]) so vai funcionar apropriadamente quando todas as matrizes forem do numpy
        #B = np.hstack((A, A[i]))

    return 0, 0

#def simplex():
#    padrao
#    tamanho da basica e n basica
#    escolher randomicamente quais colunas vao formar a matriz basica e n basica


if __name__ == "__main__":
    """A = [[1, 2, 3], [3, 1, -1], [0, 4, 2]]
    identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    result = cofactor_expansion(A);
    print(result)

    if result != 0:
        inverse = inverse(A, identity)
        for line in inverse:
            print([round(element, 4) for element in line])
    else:
        print("Determinante é 0, logo não existe inversa")
    """

    try:
        tipo, c, A, b, sinals = read_txt('func.txt')
        
        print(f"Tipo: {tipo}")
        print(f"função objetivo: {c}")
        print("Matriz A (restrições):")
        for i in range(len(A)):
            print(f"  {A[i]} {sinals[i]} {b[i]}")
            
    except FileNotFoundError:
        print("Crie um arquivo 'func.txt' na mesma pasta para testar.")

    A, c = normalized_func(A, sinals, c)
    print(f"matriz A:\n{A}")
    print(f"matriz b:\n{b}")
    print(f"matriz c:\n{c}")

    if tipo == "max":
        c = all_min(c)

    B, NB = basic_nonBasic(A, b)
    print(f"\n\n\nmatriz básica: {B}\n\nmatriz não básica {NB}")
    print(B, NB)
