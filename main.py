# Trabalho para matéria de Modelagem e Otimização Algoritimica
# Alunos: João Augusto da Silva Gomes e Vitor Felipe de Souza Siqueira
# Professor: Mateus Filipe Tavares Carvalho

# Implementação do algoritmo Simplex para identificação da solução
# ótima de um problema de programação linear de minimização

import numpy as np

def simplex(func_obj, restricoes, var_base):
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
    print("Passo 1: calcular a solução básica atual\n")
    b = np.array([restricoes[i][-1] for i in range(len(restricoes))])
    Binv = np.linalg.inv(base)
    Xb = np.matmul(Binv, b)
    print("Xb: \n", Xb, "\n")

    # Passo 2: calcular os custos relativos

    # Passo 3: verificar se a solução atual é ótima

    # Passo 4: calcular a solução simplex

    # Passo 5: determinar variavel a sair da base

    # Passo 6: atualizar a base

    iteracoes += 1


def main():
  # arquivo de entrada?
  # calcular variaveis da base?

  # Inicialização de um problema na forma padrão
  func_obj = np.array([-3, -5, 0, 0, 0])
  restricoes = np.array([[3, 2, 1, 0, 0, 18], 
                [1, 0, 0, 1, 0, 4], 
                [0, 2, 0, 0, 1, 12]])
  var_base = np.array([2, 3, 4])

  simplex(func_obj, restricoes, var_base)

if __name__ == "__main__":
    main()

