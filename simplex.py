import math
import re
import numpy as np
import random

random.seed()

# define para 6 casas decimais ou 0
def clean_value(value, decimals=5):
    if abs(value) < (10 ** -decimals):
        return 0.0
    
    return round(float(value), decimals)

def clean_value(value, threshold=1e-10):
    if abs(value) < threshold:
        return 0.0
    
    return float(value)

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
            raise ValueError("Foi encontrada uma divisão por 0. Entao a matriz e singular.")

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

#normaliza as restrições - forma padrão / tambem prepara a fase 1
def normalized_func(A, b, sinals, c):
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)
    c_np = np.array(c, dtype=float)
    
    n_lines = A_np.shape[0]
    n_orig_vars = A_np.shape[1]
    
    # mult por -1 e inverte o sinal
    for i in range(n_lines):
        if b_np[i] < 0:
            b_np[i] *= -1.0
            A_np[i] *= -1.0
            if sinals[i] == '<=':
                sinals[i] = '>='
            elif sinals[i] == '>=':
                sinals[i] = '<='

    # listas para colunas de custos
    extra_columns = []
    c_extra_phase1 = []
    c_extra_phase2 = []
    
    B_idx = [] # indices de quem começa na base 
    artificial_idx = [] # guardando temp as artificial 
    
    current_col = n_orig_vars
    
    # 2. add var de folga
    for i in range(n_lines):
        sinal = sinals[i]
        
        if sinal == '<=':
            col = np.zeros(n_lines)
            col[i] = 1.0
            extra_columns.append(col)
            c_extra_phase1.append(0.0)
            c_extra_phase2.append(0.0)
            B_idx.append(current_col)
            current_col += 1
            
        # adiciona variáveis artificiais -> vetor preenchido com 0, exceto nas posições com var artificial
        elif sinal == '>=':
            col_exc = np.zeros(n_lines)
            col_exc[i] = -1.0
            extra_columns.append(col_exc)
            c_extra_phase1.append(0.0)
            c_extra_phase2.append(0.0)
            current_col += 1
            
            col_art = np.zeros(n_lines)
            col_art[i] = 1.0
            extra_columns.append(col_art)
            c_extra_phase1.append(1.0)
            c_extra_phase2.append(0.0)
            B_idx.append(current_col)
            artificial_idx.append(current_col)
            current_col += 1
            
        elif sinal == '=':
            col_art = np.zeros(n_lines)
            col_art[i] = 1.0
            extra_columns.append(col_art)
            c_extra_phase1.append(1.0)
            c_extra_phase2.append(0.0)
            B_idx.append(current_col)
            artificial_idx.append(current_col)
            current_col += 1

    # coloca nas matrizes iniciais
    if extra_columns:
        extra_matrix = np.column_stack(extra_columns)
        A_np = np.hstack((A_np, extra_matrix))
        
    # custos finais
    c_phase2 = np.concatenate((c_np, c_extra_phase2))
    c_phase1 = np.concatenate((np.zeros(n_orig_vars), c_extra_phase1))
    
    # pega não básicas
    NB_idx = [j for j in range(A_np.shape[1]) if j not in B_idx]
    
    return A_np, b_np, c_phase1, c_phase2, B_idx, NB_idx, artificial_idx

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

def simplex2(c, B, NB, B_idx, NB_idx, inverse_B, b, type):
    C_B = [clean_value(c[i]) for i in B_idx]
    C_N = [clean_value(c[i]) for i in NB_idx]

    print(f"Variáveis Básicas (índices): {B_idx} | Custos (C_B): {C_B}")
    print(f"Variáveis Não Básicas (índices): {NB_idx} | Custos (C_N): {C_N}")

    NB_list = NB.tolist() if isinstance(NB, np.ndarray) else NB
    inverse_B_list = inverse_B.tolist() if isinstance(inverse_B, np.ndarray) else inverse_B

    pi_matrix = multiply([C_B], inverse_B_list)
    pi = [clean_value(val) for val in pi_matrix[0]] 
    print(f"\nvalor do vetor multiplicador: {pi}")

    pi_N_matrix = multiply([pi], NB_list)
    pi_N = pi_N_matrix[0]
    
    ralative_CN = []
    for i in range(len(C_N)):
        value = C_N[i] - pi_N[i]
        ralative_CN.append(clean_value(value))
        
    print(f"Custos Relativos (ralative_CN): {ralative_CN}")

    min_cost = min(ralative_CN)
    
    if min_cost >= 0:
        print("\nTodos os custos são maiores que 0 entao a solução ta ótima")
        b_matrix = [[val] for val in b]
        xB_matrix = multiply(inverse_B_list, b_matrix)
        final_sol = [clean_value(line[0]) for line in xB_matrix]

        Z_matrix = multiply([C_B], xB_matrix)
        Z = clean_value(Z_matrix[0][0])
        
        if type == "max":
            Z = Z * -1
            Z = clean_value(Z)

        total_vars = len(c)
        full_solution = [0.0] * total_vars
        
        for i in range(len(B_idx)):
            posicao_real = B_idx[i]
            full_solution[posicao_real] = final_sol[i]

        full_solution = [round(val, 5) for val in full_solution]
        Z = round(Z, 5)

        print(f"Solução Completa (os valores do vetor x): {full_solution}")
        print(f"Solução ótima (Z): {Z}")

        return True, None, None
    
    enter_idx = ralative_CN.index(min_cost) #posição do custo negativo no vetor
    
    new_col = NB_idx[enter_idx]
    print(f"\no menor custo é {min_cost}. A variável original (coluna {new_col}) entra")

    a_Nk = [[line[enter_idx]] for line in NB_list] #pega a col da variavel que entra
    y_matrix = multiply(inverse_B_list, a_Nk) #calc a direção do simplex
    y = [clean_value(line[0]) for line in y_matrix]
    print(f"Direção Simplex (y): {y}")

    b_matrix = [[val] for val in b]
    xB_matrix = multiply(inverse_B_list, b_matrix) #calc valor atual das variaveis de x chapeu
    relative_xB = [clean_value(line[0]) for line in xB_matrix]
    print(f"Solução Básica Atual (relative_xB): {relative_xB}")

    min_fact = float('inf')
    leave_idx = -1

    for i in range(len(y)): # dividindo para ver quem sai da base
        if y[i] > 0:
            fact = relative_xB[i] / y[i]
            if fact < min_fact:
                min_fact = fact
                leave_idx = i
    
    if leave_idx == -1:
        print("\nTodos os valores de y são <= 0. problema infinito")
        return True, None, None
        
    out_col = B_idx[leave_idx]
    print(f"A variável original (coluna {out_col}) sai\nRazão = {min_fact}).")

    return False, enter_idx, leave_idx

def remove_artificial(NB_idx, artificial_idx):
    # retorna uma nova lista contendo apenas quem n for artificial
    new_NB_idx = [idx for idx in NB_idx if idx not in artificial_idx]
    return new_NB_idx

if __name__ == "__main__": # 385, 520
    try:
        type, c_original_input, A_input, b_input, sinals = read_txt('func.txt')
    except FileNotFoundError:
        print("Crie um arquivo 'func.txt' na mesma pasta para testar.")
        exit()

    # ATENÇÃO: A nova função recebe 'b' e retorna mais variáveis
    A, b, c_phase1, c_phase2, B_idx_F1, NB_idx_F1, artificial = normalized_func(A_input, b_input, sinals, c_original_input)
    
    print("\n\nMATRIZES DEPOIS DE NORMALIZADAS:\n")
    print(f"matriz A:\n{A}")
    print(f"matriz b:\n{b}")
    print(f"c original:\n{c_phase2}")
    if len(artificial) > 0:
        print(f"c Fase 1:\n{c_phase1}")

    if type == "max":
        c_phase2 = all_min(c_phase2) # Inverte só o da fase 2. O da fase 1 já é de minimização!

    max_it = 20

    # ve se vai usar a fase 1
    if len(artificial) == 0:
        print("\n=== N tem variaveis artificiais ===")
        print("vai direto para a fase 2")
        
        tested_bases = set() 
        while True:
            B_np, NB_np, s_vals, nb_vals = basic_nonBasic(A, b)
            base_tuple = tuple(sorted(s_vals))
            if base_tuple in tested_bases:
                continue 
            tested_bases.add(base_tuple)
            
            result = cofactor_expansion(B_np)
            if clean_value(result) != 0.0:
                identity_temp = gen_identity(B_np)
                inverse_temp = inverse(B_np, identity_temp)
                b_matrix = [[val] for val in b]
                xB_matrix = multiply(inverse_temp, b_matrix)
                x_B_temp = [clean_value(line[0]) for line in xB_matrix]
                
                its_fact = True
                for val in x_B_temp:
                    if val < 0:
                        its_fact = False
                        break
                if its_fact:
                    print(f"Base Factível Encontrada no sorteio! xB: {x_B_temp}")
                    new_B, new_NB = B_np, NB_np
                    sorted_values, nb_values = s_vals, nb_vals
                    inverse_B = inverse_temp 
                    break

        # aqui começa a fase 2
        great, it = False, 1
        while not great and it <= max_it:
            print(f"\n>>> ITERAÇÃO {it} <<<")
            great, enter_idx, leave_idx = simplex2(c_phase2, new_B, new_NB, sorted_values, nb_values, inverse_B, b, type)
            if not great:
                enter_col, leave_col = nb_values[enter_idx], sorted_values[leave_idx]
                sorted_values[leave_idx] = enter_col
                nb_values[enter_idx] = leave_col
                for i in range(len(sorted_values)): new_B[:, i] = A[:, sorted_values[i]]
                for i in range(len(nb_values)): new_NB[:, i] = A[:, nb_values[i]]
                inverse_B = inverse(new_B, gen_identity(new_B))
                it += 1

    else:
        print("monta base inicial e inicia fase 1")
        
        # constrói as matrizes usando os indices que a normalized_func gerou 
        n_lines = len(A)
        new_B = np.zeros((n_lines, n_lines), dtype=float)
        new_NB = np.zeros((n_lines, len(NB_idx_F1)), dtype=float)
        
        for i in range(len(B_idx_F1)): new_B[:, i] = A[:, B_idx_F1[i]]
        for i in range(len(NB_idx_F1)): new_NB[:, i] = A[:, NB_idx_F1[i]]
            
        inverse_B = inverse(new_B, gen_identity(new_B))
        sorted_values = B_idx_F1.copy()
        nb_values = NB_idx_F1.copy()
        
        great, it = False, 1
        
        # loop da fase 1
        while not great and it <= max_it:
            print(f"\n>>> ITERAÇÃO FASE I - {it} <<<")
            # passa c_phase1 (vetor que só tem 1 nas artificiais) e type=min -> fase 1 tem q sempre minimizar
            # o simplex calcula o pi usando esse vetor de custo "falso"
            great, enter_idx, leave_idx = simplex2(c_phase1, new_B, new_NB, sorted_values, nb_values, inverse_B, b, "min")
            if not great:
                enter_col, leave_col = nb_values[enter_idx], sorted_values[leave_idx]
                sorted_values[leave_idx] = enter_col
                nb_values[enter_idx] = leave_col
                for i in range(len(sorted_values)): new_B[:, i] = A[:, sorted_values[i]]
                for i in range(len(nb_values)): new_NB[:, i] = A[:, nb_values[i]]
                inverse_B = inverse(new_B, gen_identity(new_B))
                it += 1
                
        # ve se deu certo -> se ainda tiver artificial não zerada, da errado
        b_matrix = [[val] for val in b]
        xB_matrix = multiply(inverse_B, b_matrix)
        Z_phase1_matrix = multiply([[clean_value(c_phase1[i]) for i in sorted_values]], xB_matrix)
        Z_phase1 = clean_value(Z_phase1_matrix[0][0])
        
        if Z_phase1 > 0:
            print(f"\nDeu errado -> A Fase I encerrou com Z = {Z_phase1}.")
            print("Isso significa que o problema n tem solução")
        else:
            print("\nZ da fase I = 0.\nComeçando fase 2")
            
            # exclui os indices das var artificiais da nb_values e depois recria a nb_values
            # após isso a fase 1 acaba e se inicia a fase 2
            nb_values = remove_artificial(nb_values, artificial)
            new_NB = np.zeros((n_lines, len(nb_values)), dtype=float)
            for i in range(len(nb_values)): new_NB[:, i] = A[:, nb_values[i]]
            
            # exec fase 2
            great, it = False, 1
            while not great and it <= max_it:
                print(f"\n>>> ITERAÇÃO FASE II - {it} <<<")
                great, enter_idx, leave_idx = simplex2(c_phase2, new_B, new_NB, sorted_values, nb_values, inverse_B, b, type)
                if not great:
                    enter_col, leave_col = nb_values[enter_idx], sorted_values[leave_idx]
                    sorted_values[leave_idx] = enter_col
                    nb_values[enter_idx] = leave_col
                    for i in range(len(sorted_values)): new_B[:, i] = A[:, sorted_values[i]]
                    for i in range(len(nb_values)): new_NB[:, i] = A[:, nb_values[i]]
                    inverse_B = inverse(new_B, gen_identity(new_B))
                    it += 1

    print("\n##### exec terminou ######################################################################")

    # Usar exercício 5.4 da página 59 para auxílio -> exercício que estou lendo no arquivo
