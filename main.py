# Trabalho para matéria de Modelagem e Otimização Algoritimica
# Alunos: João Augusto da Silva Gomes e Vitor Felipe de Souza Siqueira
# Professor: Mateus Filipe Tavares Carvalho

# Implementação do algoritmo Simplex para identificação da solução
# ótima de um problema de programação linear de minimização ou maximização

import numpy as np
import sys

# imprime uma função objetivo ou uma solução
def printFunc(nome, func):
  print(f"{nome}: ", end='')
  for i in range(len(func)):
    if i == len(func)-1:
      print("x{} = {}".format(i+1, func[i]))
    else:
      print("x{} = {}".format(i+1, func[i]), end=', ')


# função para leitura do arquivo de entrada e transformação do problema em forma padrao
def forma_padrao(entrada):
  with open(entrada, 'r') as arq:
    func_obj = []
    restricoes = []
    num_linha = 0
    var_adicionais = []
    igualdades = []
    irrestritas = []
    positivas = []
    negativas = []
    sobras = []

    for linha in arq.readlines():
        linha = linha.strip()
        print(func_obj)

        # le a função objetivo min
        if linha.startswith('min'):
          aux = linha.split(' ');
          func_obj.append(float(aux[1]))
          for i in range(3, len(aux)):
            if aux[i].isdigit():
              func_obj.append(float(aux[i-1] + aux[i]))

        # le a função objetivo max
        elif linha.startswith('max'):
          aux = linha.split(' ');
          func_obj.append(-float(aux[1]))
          for i in range(3, len(aux)):
            if aux[i].isdigit():
              func_obj.append(-float(aux[i-1] + aux[i]))

        # le as restrições
        elif linha.startswith('s.a'):
          aux = linha.split(' ')
          restricao = []
          restricao.append(float(aux[1]))
          igualdades.append(float(aux[-1]))
          for i in range(3, len(aux)):
            if aux[i].isnumeric and (aux[i-1] == '+' or aux[i-1] == '-'):
              restricao.append(float(aux[i-1] + aux[i]))
            elif aux[i] == '<=' or aux[i] == '>=' or aux[i] == '<' or aux[i] == '>':
              if aux[i] == '>=':
                var_adicionais.append([num_linha, False])  
              else:
                var_adicionais.append([num_linha, True])
              func_obj.append(0)  # adiciona variáveis adicionais na função objetivo
          restricoes.append(restricao)
          num_linha += 1

        # le as variaveis com restricoes >= L , L pertencendo aos reais
        elif not linha.endswith('0') and not linha.endswith('livre'):
          aux = linha.split(' ')
          var = int(aux[0][1:])-1
          if var not in range(len(restricoes[0])):
            print("Variável x{} não existe.".format(var+1))
            sys.exit(1)
          positivas.append([var, float(aux[2])])

        # le as variaveis irrestritas
        elif linha.endswith('livre'):
          aux = linha.split(' ')
          var = int(aux[0][1:])-1
          if var not in range(len(restricoes[0])):
            print("Variável x{} não existe.".format(var+1))
            sys.exit(1)
          irrestritas.append(var)
        
        # le as variaveis com restricoes <= 0
        elif linha.endswith('0'):
          aux = linha.split(' ')
          if not aux[1] == '<=':
            break
          print(linha)
          var = int(aux[0][1:])-1
          if var not in range(len(restricoes[0])):
            print("Variável x{} não existe.".format(var+1))
            sys.exit(1)
          negativas.append(var)

    # adiciona sobras na função objetivo
    for i in range(len(positivas)):
      for j in range(len(func_obj)):
        if positivas[i][0] == j:
          sobras.append(positivas[i][1]*func_obj[j])

    # adiciona variáveis irrestriras na função objetivo
    for i in range(len(irrestritas)):
      for j in range(len(func_obj)):
        if irrestritas[i] == j:
          func_obj.insert(j+1, -func_obj[j])

    # adiciona as variáveis adicionais na matriz de restrições
    for i in range(len(var_adicionais)):
      for j in range(len(restricoes)):
        if var_adicionais[i][0] == j:
          if var_adicionais[i][1]:
            restricoes[j].append(1)
          else:
            restricoes[j].append(-1)
        else:
          restricoes[j].append(0)

    # adiciona as igualdades na matriz de restrições
    for i in range(len(igualdades)):
      restricoes[i].append(igualdades[i])

    # se a igualdade for negativa, inverte o sinal da restrição
    for i in range(len(restricoes)):
      if restricoes[i][-1] < 0:
        for j in range(len(restricoes[i])):
          restricoes[i][j] = -restricoes[i][j] 

    # adiciona as variáveis positivas na matriz de restrições
    for i in range(len(positivas)):
      for linha in restricoes:
        for j in range(len(linha)):
          if positivas[i][0] == j:
            linha[-1] = linha[-1] - positivas[i][1]*linha[j]
    
    # adiciona as variáveis negativas na matriz de restrições
    for i in range(len(negativas)):
      for linha in restricoes:
        for j in range(len(linha)):
          if negativas[i] == j:
            linha[j] = -linha[j]

    # adiciona as variáveis irrestritas na matriz de restrições
    for i in range(len(irrestritas)):
      for linha in restricoes:
        for j in range(len(linha)):
          if irrestritas[i] == j:
            linha.insert(j+1, -linha[j])

    return {
      "func_obj": func_obj,
      "restricoes": restricoes,
      "sobras": sobras,
    }


# função para identificar as variáveis da base
def vars_decisao(restricoes):
  variaveis = []
  for i in range(len(restricoes[0])-1):
    coluna = restricoes[:,i]
    if np.count_nonzero(coluna) == 1: # and np.sum(coluna) == 1:
      variaveis.append(i)
  
  if len(variaveis) < len(restricoes):
    for i in reversed(range(len(restricoes))):
      if i not in variaveis:
        variaveis.insert(0,i)
        break

  print("Variáveis de decisão: ", end='')
  for i in range(len(variaveis)):
    if i == len(variaveis)-1:
      print(f"x{variaveis[i]+1}\n")
    else:
      print(f"x{variaveis[i]+1}", end=', ')
  return np.array(variaveis)


# função que calcula a razao minima
def calcular_razao_minima(Xb, y):
    razoes = []
    for i in range(len(Xb)):
        if y[i] > 0:
            razoes.append(Xb[i] / y[i])
        else:
            razoes.append(float('inf'))  # Adiciona infinito onde não é possível calcular a razão
    indice_menor_razao = razoes.index(min(razoes))  # Índice da menor razão válida
    return indice_menor_razao, razoes[indice_menor_razao]


def sair_base(Xb, y, var_base):
    razoes = np.array([Xb[i] / y[i] if y[i] > 0 else np.inf for i in range(len(y))])
    indice_menor_razao = np.argmin(razoes)  # Encontra o índice da menor razão

    if razoes[indice_menor_razao] == np.inf:
        print("Problema ilimitado.")
        sys.exit(1)  # Encerra o programa indicando que o problema é ilimitado

    variavel_sair = var_base[indice_menor_razao]  # Encontra a variável correspondente a sair
    
    # Imprimir a variável que sai da base, não apenas o índice
    print(f"Variável a sair da base: x{variavel_sair + 1} (Índice: {variavel_sair}, Valor: {Xb[indice_menor_razao]})")
    return variavel_sair
  

def atualizar_bases(var_base, var_nao_base, variavel_sair, variavel_entrar):
    # A função atualizar_bases atualiza as listas var_base e var_nao_base
    indice_sair = np.where(var_base == variavel_sair)[0][0]
    indice_entrar = np.where(var_nao_base == variavel_entrar)[0][0]
    
    var_base[indice_sair], var_nao_base[indice_entrar] = \
    var_nao_base[indice_entrar], var_base[indice_sair]
    return var_base, var_nao_base

  
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
  B_inv = np.linalg.inv(base)
  Xb = np.dot(B_inv, restricoes[:, -1])

  # Iterações
  iteracoes = 0
  while True:
    print("------------------------Iteração: ", iteracoes+1, "------------------------\n")
    print("B:\n", base, "\n")
    print("N:\n", nao_base, "\n")
    print("CBT: ", cbt, "\n")
    print("CNT: ", cnt, "\n")
    print("var_base: ", var_base, "\n")

    # Passo 1: calcular a solução básica atual
    print("///////// Passo 1: calcular a solução básica atual\n")
    print("X = Binv . Xb\n")
    b = np.array([restricoes[i][-1] for i in range(len(restricoes))])
    print("b: ", b, "\n")
    try: # Verifica se a matriz base possui inversa
      Binv = np.linalg.inv(base)
    except np.linalg.LinAlgError:
      print("A matriz base não possui inversa.")
      sys.exit(1)
    Xb = np.matmul(Binv, b)
    print("Binv:", Binv, "\n")
    print("Xb:", Xb, "\n")

    # Passo 2: calcular os custos relativos
    print("///////// Passo 2: calcular os custos relativos\n")
    print("i- λt = Cbt . Binv\nii- Cn = Cnj - λt . anj \niii- Cnk = min (Cnj)\n")
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
    print("Cn[k] >= 0\n")
    if Cn[k] >= 0:
      print("Cnk >= 0, solução é ótima\n")
      solucao = [0 for i in func_obj]
      Xb_indice = 0
      for i in var_base:
        solucao[i] = Xb[Xb_indice]
        Xb_indice += 1
      for i in var_nao_base:
        solucao[i] = 0
      return solucao
    else:
      print("Cnk < 0, solução não é ótima\n")
      
    # Selecione a coluna k para entrar na base
    k = np.argmin(Cn)
    

    # Passo 4: calcular a solução simplex
    print("///////// Passo 4: calcular o simplex")
    print("y = Binv . ank\n")
    ak = nao_base[:, k]
    print("Base B atual:\n", base)
    print("\nVetor ak (coluna que entra na base):\n", ak)
    y = np.linalg.solve(base, ak)
    print("\nVetor y (solução simplex):\n", y, "\n")
    
    
    # Passo 5: determinar variavel a sair da base
    print("///////// Passo 5: determinar a variável a sair da base\n")
    print("Ê = min {Xb[i] / y[i]}\n")
    razoes = np.array([Xb[i] / y[i] if y[i] > 0 else np.inf for i in range(len(y))])
    print("Matriz das razões:\n", razoes)
    l = np.argmin(razoes)
    if razoes[l] == np.inf:
        print("O problema é ilimitado.")
        sys.exit(1)
        
    variavel_sair = var_base[l]
    print(f"\nVariável a sair da base: x{variavel_sair + 1}, Variável a entrar na base: x{var_nao_base[k] + 1}\n")

   
    # Passo 6: atualização
    variavel_sair = var_base[l]  # Variável que sai da base
    var_base, var_nao_base = atualizar_bases(var_base, var_nao_base, variavel_sair, var_nao_base[k])  # Atualiza as variáveis de base e não base

    # Recalcula a base e a não base para a próxima iteração
    base = restricoes[:, var_base].astype(float)
    nao_base = restricoes[:, var_nao_base].astype(float)
    cbt = np.array([func_obj[i] for i in var_base])
    cnt = np.array([func_obj[i] for i in var_nao_base])

    # Recalcula a inversa da nova base
    B_inv = np.linalg.inv(base)
    
    # Recalcula a solução básica atual
    Xb = np.dot(B_inv, restricoes[:, -1])

    iteracoes += 1



def main():
  if len(sys.argv) < 2:
    print("Uso: python3 main.py <arquivo de entrada>")
    return
  
  sys.stdout = open('output.txt', 'w')
  sys.stdout.reconfigure(encoding='utf-8')
  
  ppl = forma_padrao(sys.argv[1])

  printFunc("Função Objetivo", ppl.get("func_obj"))

  print("Restrições:\n", np.array(ppl.get("restricoes")))

  vars_solucao = simplex(np.array(ppl.get("func_obj")), np.array(ppl.get("restricoes")))

  printFunc("Variaveis solução", vars_solucao)

  print("Variáveis de sobra: ", ppl.get("sobras"))
  solucao = 0
  for i in ppl.get("func_obj"):
    solucao += i * vars_solucao[ppl.get("func_obj").index(i)]
  for i in ppl.get("sobras"):
    solucao += i
  print("Solução:", solucao)


if __name__ == "__main__":
    main()