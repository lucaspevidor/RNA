import numpy as np
from matplotlib import pyplot as plt

class Neuronio:

    #Função de inicialização da classe
    def __init__(self, n_entradas, func_act, use_bias):
        #Se optar por utilizar bias, será adicionada uma coluna no final dos vetores de entrada com valor unitário.
        self.use_bias = use_bias        
        if use_bias:
            self.pesos = np.empty((n_entradas+1, 1), dtype=float)
        else:
            self.pesos = np.empty((n_entradas, 1), dtype=float)
        self.func_act = func_act
    
    def reset(self):
        #Zerar pesos
        self.pesos = np.zeros((len(self.pesos[:,0]), 1), dtype=float)
            
    #Definição de entradas
    #Função utilizada para configurar o vetor de entrada interno da classe, juntamente com normalização, adição
    #da coluna de bias. Utilizar quando a média e desvpad para normalização ainda não foram definidos.
    def set_entradas_ant(self, entradas, normalizar):        
        if self.use_bias:
            vetor_1 = np.zeros([len(entradas[:,0]), 1]) + 1
            if normalizar:
                self.normalizar(entradas, False)
                self.vetor_entradas = np.append(self.entr_normalizada, vetor_1, axis=1)
            else:
                self.vetor_entradas = np.append(entradas, vetor_1, axis=1)
        else:
            if normalizar:                
                self.vetor_entradas = self.normalizar(entradas, False)
            else:
                self.vetor_entradas = entradas
    
    #Definir entradas após treinamento da rede
    #Função utilizada para configurar o vetor de entrada interno da classe, juntamente com normalização, adição
    #da coluna de bias. Utilizar quando a média e desvpad para normalização ainda JÁ foram definidos.
    def set_entradas_pós(self, entradas, normalizar):
        if self.use_bias:
            vetor_1 = np.zeros([len(entradas[:,0]), 1]) + 1
            if normalizar:            
                self.vetor_entradas = np.append(self.normalizar(entradas, True), vetor_1, axis=1)
            else:
                self.vetor_entradas = np.append(entradas, vetor_1, axis=1)
        else:
            if normalizar:
                self.vetor_entradas = self.normalizar(entradas, True)
            else:                    
                self.vetor_entradas = entradas
    
    def normalizar(self, entradas, utilizar_parametros_anteriores):
        if utilizar_parametros_anteriores == False:
            #Encontrar média e desvio padrão a partir das entradas fornecidas.
            #Executado antes do treinamento da rede
            self.media = np.empty(len(entradas[0,:]), dtype=float)
            self.desvpad = np.empty(len(entradas[0,:]), dtype=float)
            self.entr_normalizada = np.empty((len(entradas[:,0]), len(entradas[0,:])), dtype=float)
        
            for i in range(0, len(entradas[0,:])):
                self.media[i] = np.mean(entradas[:,i])
                self.desvpad[i] = np.sqrt(np.var(entradas[:,i]))            
                self.entr_normalizada[:,i] = (entradas[:,i] - self.media[i] ) / self.desvpad[i]
            return self.entr_normalizada
        else:
            #Normaliza os dados de entrada fornecidos conforme a média e desvio padrão já encontrados anteriormente
            entr_nova_normalizada = np.empty((len(entradas[:,0]), len(entradas[0,:])), dtype=float)
            for i in range(0, len(entradas[0,:])):                
                entr_nova_normalizada[:,i] = (entradas[:,i] - self.media[i] ) / self.desvpad[i]            
            
            return entr_nova_normalizada    
        
    def proc_saida(self, entradas, normalizado):
        if not normalizado:
            #Normalização das entradas utilizando média e desvpad já encontrados
            entr_nova_normalizada = self.normalizar(entradas, True)
        else:
            entr_nova_normalizada = entradas
        
        v = np.dot(entr_nova_normalizada, self.pesos)
        saida = np.empty([1,1])
        if(self.func_act == 'Bipolar'):    
            for i in range(len(v)):
                if(v[i,:] > 0):
                    saida = np.append(saida, [[1]], axis=0)
                else:
                    saida = np.append(saida, [[-1]], axis=0)
            self.saida = saida[1:]
            return saida[1:]
        
        elif(self.func_act == 'Linear'):
            saida = v
            return saida

        elif(self.func_act == 'Sigmoid'):
            saida = 1/(1+np.e**(-v))
            return saida
        
            
    def treinar_rede(self, fator_aprendizado, saida_desejada, max_iterações):
        redeTreinada = False
        self.historicoAcertos = []
        self.historicoPesos = np.empty([len(self.pesos), 1])
        
        for i in range(0, max_iterações):
            if not redeTreinada:
                flagAlteração = False
                acertos = 0
                for j in range(len(self.vetor_entradas[:,0])):
                    #Testa as entradas até encontrar um erro
                    #Caso encontrado, continua testando as outras entradas para obter a quantidade de acertos para
                    #os pesos atuais. Em seguida atualiza os pesos e reinicia os testes.
                    saida_proc = self.proc_saida([self.vetor_entradas[j,:]], True)[0,0]
                    res = saida_desejada[j,0] - saida_proc
                    if res == 0 and self.func_act=='Bipolar':
                        acertos += 1
                    else:
                        if self.func_act == 'Linear' or self.func_act == 'Bipolar':
                            deltaW = fator_aprendizado * res * self.vetor_entradas[j,:]
                        elif self.func_act == 'Sigmoid':
                            deltaW = fator_aprendizado * res * self.vetor_entradas[j,:] * saida_proc * (1 - saida_proc)
                        flagAlteração = True                                                
                        
                        #Se a rede for bipolar, termina de checar a quantidade de acertos
                        #antes de atualizar os pesos
                        if self.func_act=='Bipolar':
                            for k in range(j+1, len(self.vetor_entradas[:,0])):
                                res = saida_desejada[k,0] - self.proc_saida([self.vetor_entradas[k,:]], True)[0,0]
                                if res == 0:
                                    acertos += 1
                        
                            #Salva os acertos e pesos
                            self.historicoAcertos.append(acertos)
                        self.historicoPesos = np.append(self.historicoPesos, self.pesos, axis=1)
                        
                        #Atualiza os pesos                               
                        self.pesos += np.transpose([deltaW])
                        if self.func_act == 'Bipolar':
                            break
                
                if flagAlteração == False:
                    redeTreinada = True
                    self.historicoAcertos.append(acertos)
                    self.historicoPesos = np.append(self.historicoPesos, self.pesos, axis=1)                
        
        self.historicoErros = []
        for i in range(len(self.historicoAcertos)):            
            self.historicoErros.append(len(self.vetor_entradas) - self.historicoAcertos[i])
        
        if redeTreinada == False:
            if self.func_act=='Bipolar':
                #Se não foi possível obter 100% de acerto durante as iterações
                #encontra e define como pesos os que obtiveram mais acertos
                indiceMax = self.historicoAcertos.index(max(self.historicoAcertos))                        
                self.pesos = np.transpose([self.historicoPesos[:, indiceMax+1]])
                print('Nº erros: {}'.format(len(self.vetor_entradas[:,0])-max(self.historicoAcertos)))

class Camada:
    def __init__(self, n_entradas, n_neuronios, func_act):
        self.n_neuronios = n_neuronios
        self.n_entradas = n_entradas
        self.func_act = func_act
        self.saidas = np.zeros([n_neuronios, 1])
        self.entradas = np.zeros([n_entradas, 1])

        self.neuronios = []
        for i in range(n_neuronios):
            self.neuronios.append(Neuronio(n_entradas, func_act, use_bias=True))
    
    def setEntradas(self):
        for i in range(self.n_neuronios):
            self.neuronios[i].set_entradas_ant(self.entradas, False)

    def proc_saida(self):
        self.setEntradas()
        for i in range(self.n_neuronios):
            self.saidas[i,0] = self.neuronios[i].proc_saida(self.neuronios[i].vetor_entradas, normalizado=True)

class RedeNeural:
    def __init__(self, n_entradas, n_camadasOcultas, nn_camadasOcultas, n_saidas, func_act):
        self.n_entradas = n_entradas
        self.n_camadasOcultas = n_camadasOcultas
        self.n_saidas = n_saidas
        self.func_act=func_act
        self.nn_camadasOcultas = nn_camadasOcultas
        
        self.entradas = np.zeros([n_entradas, 1])
        self.saidas = np.zeros([n_saidas, 0])

        self.camadas = []
        #Primeira camada, 1 neurônio para cada entrada
        c = 0
        self.camadas.append(Camada(n_entradas, n_entradas, func_act))
        self.camadas[c].entradas = self.entradas
        c += 1
        #Primeira camada oculta, n_entradas igual a nº de entradas
        if n_camadasOcultas != 0:
            self.camadas.append(Camada(n_entradas, nn_camadasOcultas, func_act))
            self.camadas[c].entradas = self.camadas[c-1].saidas
            c += 1
            #Demais camadas ocultas
            for i in range(n_camadasOcultas-1):
                self.camadas.append(Camada(nn_camadasOcultas, nn_camadasOcultas, func_act))
                self.camadas[c].entradas = self.camadas[c-1].saidas
                c += 1
        self.camadas.append(Camada(nn_camadasOcultas, n_saidas, func_act))
        self.camadas[c].entradas = self.camadas[c-1].saidas

        self.n_camadas = c+1

R1 = RedeNeural(2, 3, 5, 1, 'Sigmoid')
print('ok')