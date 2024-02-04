# Trabalho para matéria de Modelagem e Otimização Algoritimica
# Alunos: João Augusto da Silva Gomes e Vitor Felipe de Souza Siqueira
# Professor: Mateus Filipe Tavares Carvalho

# Implementação do algoritmo Simplex para identificação da solução
# ótima de um problema de programação linear de minimização

import numpy as np
import sys

# função para leitura do arquivo de entrada e transformação do problema em forma padrao
def forma_padrao(entrada):
  with open(entrada, 'r') as arq:
    func_obj = []
    restricoes = []
    num_linha = 0
    var_adicionais = []
    igualdades = []

    for linha in arq.readlines():
        linha = linha.strip()

        # le a função objetivo
        if linha.startswith('min'):
          aux = linha.split(' ');
          func_obj.append(-int(aux[1]))
          for i in range(3, len(aux)):
            if aux[i].isdigit():
              func_obj.append(-int(aux[i-1] + aux[i]))

        # le as restrições
        elif linha.startswith('s.a'):
          aux = linha.split(' ')
          restricao = []
          restricao.append(int(aux[1]))
          for i in range(3, len(aux)):
            if aux[i].isdigit() and (aux[i-1] == '+' or aux[i-1] == '-'):
              restricao.append(int(aux[i-1] + aux[i]))
            elif aux[i] == '<=' or aux[i] == '>=':
              var_adicionais.append(num_linha)
              func_obj.append(0)  # adiciona variáveis adicionais na função objetivo
            elif aux[i] != '=':
              igualdades.append(int(aux[i]))
          restricoes.append(restricao)
          num_linha += 1
    
    # adiciona as variáveis adicionais na matriz de restrições
    for i in range(len(var_adicionais)):
      for j in range(len(restricoes)):
        if var_adicionais[i] == j:
          restricoes[j].append(1)
        else:
          restricoes[j].append(0)

    # adiciona as igualdades na matriz de restrições
    for i in range(len(igualdades)):
      # se a igualdade for negativa, inverte o sinal da restrição
      if igualdades[i] < 0:
        for j in range(len(restricoes[i])):
          restricoes[i][j] = -restricoes[i][j]
        restricoes[i].append(-igualdades[i])
      else:
        restricoes[i].append(igualdades[i])
      
    return {
      "func_obj": func_obj,
      "restricoes": restricoes
    }

# função para identificar as variáveis da base
def vars_decisao(restricoes):
  variaveis = []
  for i in range(len(restricoes[0])):
    coluna = restricoes[:,i]
    if np.count_nonzero(coluna) == 1 and np.sum(coluna) == 1:
      variaveis.append(i)
  
  if len(variaveis) < len(restricoes):
    for i in range(len(restricoes)):
      if i not in variaveis:
        variaveis.append(i)
        break
  
  print("Variáveis de decisão: ", variaveis)
  return np.array(variaveis)

# função para execução do algoritmo simplex
def simplex(func_obj, restricoes):
  # Inicializar as variáveis de decisão
  var_base = vars_decisao(restricoes)

  # Inicializar as variáveis não base
  var_nao_base = np.array([i for i in range(len(func_obj)) if i not in var_base])

  # Inicializar a base
  base = []
  for i in range(len(restricoes)):
      for j in range(len(var_base)):
         base.append(restricoes[i][var_base[j]])
  base = np.array(base).reshape(len(restricoes), len(var_base))

  # Inicializar a não base
  nao_base = []
  for i in range(len(restricoes)):
      for j in range(len(var_nao_base)):
          nao_base.append(restricoes[i][var_nao_base[j]])
  nao_base = np.array(nao_base).reshape(len(restricoes), len(var_nao_base))

  cbt = np.array([func_obj[i] for i in var_base])
  cnt = np.array([func_obj[i] for i in var_nao_base])

  # Iterações
  iteracoes = 0
  iteracoes_max = 1
  while(iteracoes < iteracoes_max):
    print("------------Iteração: ", iteracoes+1, "\n")
    print("B:\n", base, "\n")
    print("N:\n", nao_base, "\n")
    print("CBT: ", cbt, "\n")
    print("CNT: ", cnt, "\n")

    # Passo 1: calcular a solução básica atual
    print("///////// Passo 1: calcular a solução básica atual\n")
    b = np.array([restricoes[i][-1] for i in range(len(restricoes))])
    print("b: ", b, "\n")
    Binv = np.linalg.inv(base)
    Xb = np.matmul(Binv, b)
    print("Xb:", Xb, "\n")

    # Passo 2: calcular os custos relativos
    print("///////// Passo 2: calcular os custos relativos\n")
      # i)
    lambdaT = np.matmul(cbt, Binv)
    print("lambdaT: ", lambdaT, "\n")
      # ii)
    Cn = np.array([cnt[i] - np.matmul(lambdaT, nao_base[:,i]) for i in range(len(nao_base[0]))])
    print("Cn: ", Cn, "\n")
      # iii)
    k = np.argmin(Cn)
    print("k: ", k+1, "\n")
    print("coluna", k+1, "entra na base\n")

    # Passo 3: verificar se a solução atual é ótima
    print("///////// Passo 3: verificar se a solução atual é ótima\n")
    if Cn[k] >= 0:
      print("Cnk >= 0, solução é ótima\n")
      sol = []
      b_index = 0
      cnt_index = 0
      for i in range(len(func_obj)):
        if i in var_base:
          sol.append(b[b_index])
          b_index += 1
        else:
          sol.append(cnt[cnt_index])
          cnt_index += 1
      return sol
    else:
      print("Cnk < 0, solução não é ótima\n")

    # Passo 4: calcular a solução simplex

    # Passo 5: determinar variavel a sair da base

    # Passo 6: atualização

    iteracoes += 1

def main():
  if len(sys.argv) < 2:
    print("Uso: python3 main.py <arquivo de entrada>")
    return
  ppl = forma_padrao(sys.argv[1])
  print(ppl)

  # # Inicialização de um problema na forma padrão
  # func_obj = np.array([-3, -5, 0, 0, 0])
  # restricoes = np.array([[3, 2, 1, 0, 0, 18], 
  #               [1, 0, 0, 1, 0, 4], 
  #               [0, 2, 0, 0, 1, 12]])
  # var_base = np.array([2, 3, 4])

  solucao = simplex(np.array(ppl.get("func_obj")), np.array(ppl.get("restricoes")))
  print("Solução: ", solucao)

if __name__ == "__main__":
    main()

