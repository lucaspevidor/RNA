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
        self.novosPesos = np.array([])
        self.pesosAnteriores = self.pesos

        self.func_act = func_act
        self.delta = 0        
    
    def reset(self):
        #Fazer pesos randomizados
        vetor = np.ones((len(self.pesos[:,0]), 1), dtype=float)
        for i in range(len(vetor[:,0])):
            vetor[i,0] = np.random.normal(scale=0.3)
        self.pesos = vetor
        self.pesosAnteriores = vetor
        self.novosPesos = np.array([])
    
    @staticmethod
    def fSigmoid(v):
        saida = 1/(1+np.e**(-v))
        return saida
    
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
            self.saida = v
            return saida

        elif(self.func_act == 'Sigmoid'): 
            self.saida = self.fSigmoid(v)
            return self.saida
        
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

    def trocarPesos(self):
        self.pesosAnteriores = self.pesos
        self.pesos = self.novosPesos
        self.novosPesos = np.array([])
        

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
            self.neuronios[i].reset()
    
    def setEntradas(self):
        for i in range(self.n_neuronios):
            self.neuronios[i].set_entradas_ant(np.transpose(self.entradas), False)

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
        
        self.listaEntradas = np.zeros([0,0]) #Lista com todas as entradas a serem processadas
        self.entradas = np.zeros([n_entradas, 1]) #Vetor de entradas para jogar nas camadas
        self.saidas = np.zeros([n_saidas, 1]) #Vetor de saída p/ receber das camadas
        self.listaSaidas = []

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
        else:
            self.camadas.append(Camada(n_entradas, n_saidas, func_act))
        self.camadas[c].entradas = self.camadas[c-1].saidas
        self.saidas = self.camadas[c].saidas

        self.n_camadas = c+1
    
    #Definir entradas após treinamento da rede
    #Função utilizada para configurar o vetor de entrada interno da classe, juntamente com normalização, adição
    #da coluna de bias. Utilizar quando a média e desvpad para normalização JÁ foram definidos.
    def set_entradas_pós(self, entradas, normalizar):
        if normalizar:           
            vetorEntradas = self.normalizar(entradas, True)     
            for i in range(len(vetorEntradas)):
                self.entradas[i] = vetorEntradas[i]
        else:
            for i in range(len(entradas)):
                self.entradas[i] = entradas[i]
    
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

    def proc_saidas(self, normalizar):        
        self.listaSaidas = []
        for k in range(len(self.listaEntradas)):
            self.set_entradas_pós(np.transpose([self.listaEntradas[k]]), normalizar)
            for i in range(self.n_camadas):
                self.camadas[i].proc_saida()
            self.listaSaidas.append(self.saidas)

    def proc_saida(self, indice, normalizar):
        self.listaSaidas = []
        self.set_entradas_pós(np.transpose([self.listaEntradas[indice]]), normalizar)
        for i in range(self.n_camadas):
            self.camadas[i].proc_saida()
        self.listaSaidas.append(self.saidas)

    def treinar_rede(self, saidas_desejadas, tx_aprendizado_inicial, annealing, tx_momento, n_iteracoes):        
        #primeiro passo: Processar a saída da rede com os dados atuais. A listaEntradas deve estar normalizada e configurada
        for m in range(n_iteracoes):
            if m==10000:
                print('okok')
            tx_aprendizado = tx_aprendizado_inicial#/(1+(m/annealing))
            for l in range(len(self.listaEntradas)):
                self.proc_saida(l, False)
                print('')
                #cálculo do delta k e primeiro ajuste de pesos p/ última camada
                for n in self.camadas[self.n_camadas-1].neuronios:
                    n.delta = n.saida[0][0]*(1-n.saida[0][0])*(saidas_desejadas[l][0]-n.saida[0][0])
                    momento = tx_momento * (n.pesos - n.pesosAnteriores)
                    n.novosPesos = n.pesos + np.transpose([tx_aprendizado*n.delta*(np.append(self.camadas[self.n_camadas-2].saidas, np.array([[1]])))]) + momento
                #cálculo do delta j e ajustes de pesos para camadas ocultas e primeira camada
                for c in range(self.n_camadas-2, -1, -1):
                    for k, n in enumerate(self.camadas[c].neuronios):
                        #cálculo do somatório de wkdk da camada seguinte
                        soma = 0
                        for i, n2 in enumerate(self.camadas[c+1].neuronios):
                            soma += n2.delta * n2.pesos[k, 0]
                        #cálculo do deltaJ
                        n.delta = n.saida[0][0]*(1-n.saida[0][0])*soma
                        momento = tx_momento * (n.pesos - n.pesosAnteriores)
                        n.novosPesos = n.pesos + np.transpose([tx_aprendizado*n.delta*(np.append(self.camadas[c].entradas, np.array([[1]])))]) + momento

                #Troca os pesos dos neuronios
                for i in range(len(self.camadas)):
                    for n in self.camadas[i].neuronios:
                        n.trocarPesos()



#------------------------------------------------------------------------------------------------------------------

c1 = Camada(2, 3, 'Sigmoid')
c2 = Camada(3, 1, 'Sigmoid')
c3 = Camada(1, 1, 'Sigmoid')

c2.entradas = c1.saidas
c3.entradas = c2.saidas

entr = np.transpose([[3,2]])
c1.entradas = entr
entr[1][0] = 10
c1.saidas = np.transpose([[1,5,6]])

entr = np.array([[0,0],[0,1],[1,0],[1,1]])
#entr = np.array([[0,0],[1,1]])
sd = np.transpose(np.array([[0,1,1,0]]))
#sd = np.transpose(np.array([[0,1]]))
rede = RedeNeural(2, 0, 0, 1, 'Sigmoid')
rede.listaEntradas = rede.normalizar(entr, utilizar_parametros_anteriores=False)
rede.treinar_rede(sd, 10, 600, 0.1, 11000)

print('ok')