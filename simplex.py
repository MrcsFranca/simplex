import math
import re
import numpy as np
import random

random.seed()

# define para 6 casas decimais ou 0
def clean_value(value, decimals=6):
    if abs(value) < (10 ** -decimals):
        return 0.0
    
    return round(float(value), decimals)

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

# extrai a submatriz
def sub_matrix(A, line, column):
    return np.delete(np.delete(A, line, axis=0), column, axis=1)

# metodo da expansão dos cofatores
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

# gera a matriz identidade
def gen_identity(A):
    n_lines = len(A)
    return np.eye(n_lines, dtype=float)

# calcula a matriz inversa
def inverse(A, identity):
    n = len(A)
    matrix_A = np.copy(A)
    matrix_I = np.copy(identity)
    
    for i in range(n):
        pivotLine = i
        maxValue = math.fabs(matrix_A[i, i])

        for j in range(i + 1, n):
            if math.fabs(matrix_A[j, i]) > maxValue:
                pivotLine = j
                maxValue = math.fabs(matrix_A[j, i])

        if pivotLine != i:
            matrix_A[[i, pivotLine]] = matrix_A[[pivotLine, i]]
            matrix_I[[i, pivotLine]] = matrix_I[[pivotLine, i]]

        pivot = matrix_A[i, i]

        if clean_value(pivot) == 0.0:
            raise ValueError("Foi encontrada uma divisão por 0. Matriz singular.")

        matrix_A[i] = matrix_A[i] / pivot
        matrix_I[i] = matrix_I[i] / pivot

        for j in range(n):
            if i == j:
                continue

            multiplier = matrix_A[j, i]

            matrix_A[j] -= multiplier * matrix_A[i]
            matrix_I[j] -= multiplier * matrix_I[i]

    clean_vectorized = np.vectorize(clean_value)
    return clean_vectorized(matrix_I)

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
    sorted_values = []

    sorted_values = random.sample(range(n_columns), n_lines)
    print(f"valores sorteados {sorted_values}")

    for i in range(n_lines):
        pos = sorted_values[i]
        B[:, i] = A[:, pos]

    nb_values = [col for col in range(n_columns) if col not in sorted_values]
    nb_cols = len(nb_values)
    NB = np.zeros((n_lines, nb_cols), dtype=float)
    print(nb_values)

    for i in range(nb_cols):
        pos = nb_values[i]
        NB[:, i] = A[:, pos]


    return B, NB, sorted_values, nb_values

def simplex2():
    return 
#    padrao
#    tamanho da basica e n basica

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
        
        """
        print(f"Tipo: {tipo}")
        print(f"função objetivo: {c}")
        print("Matriz A (restrições):")
        for i in range(len(A)):
            print(f"  {A[i]} {sinals[i]} {b[i]}")
        """
            
    except FileNotFoundError:
        print("Crie um arquivo 'func.txt' na mesma pasta para testar.")

    A, c = normalized_func(A, sinals, c)
    print(f"matriz A:\n{A}")
    print(f"matriz b:\n{b}")
    print(f"matriz c:\n{c}")

    if tipo == "max":
        c = all_min(c)

    tested_bases = set() 
    new_B = None
    new_NB = None
    inverse_B = None
    while True:
        B_np, NB_np, sorted_values, nb_values = basic_nonBasic(A, b)
        print(f"\nMatrix básica:\n"B_np)
        base_tuple = tuple(sorted(sorted_values))

        if base_tuple in tested_bases:
            print(f"Combinação {base_tuple} já testada. Sorteando novamente...")
            continue 
            
        tested_bases.add(base_tuple)
        print(f"\nTestando nova base com as colunas: {sorted_values}")

        result = cofactor_expansion(B_np)
        result = clean_value(result)

        if result != 0.0:
            print(f"Determinante de B = {result}. Matriz válida encontrada!")
            new_B = B_np
            new_NB = NB_np
            break
        else:
            print("Determinante de B = 0. Matriz singular. Sorteando novas colunas...")

    identity = gen_identity(new_B)
    inverse_B = inverse(new_B, identity)

    print(inverse_B)

