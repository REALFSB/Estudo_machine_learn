import gym
import random
import numpy as np
from IPython.display import clear_output
import time

# Inicialização do ambiente e exibição inicial
start_time = time.time()
env = gym.make('Taxi-v3', render_mode="ansi")
env.reset()
print(env.render())

# Informações sobre o espaço de ações e observações
print(env.action_space)
print(env.observation_space)

# Inicialização da tabela Q com zeros
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Parâmetros do algoritmo Q-learning
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# Treinamento do agente
for i in range(10000):
    estado, _ = env.reset()
    penalidades, recompensa = 0, 0
    done = False

    while not done:
        # Escolha da ação baseada em exploração ou exploração
        if random.uniform(0, 1) < epsilon:
            acao = env.action_space.sample()  # Exploração: escolhe uma ação aleatória
        else:
            acao = np.argmax(q_table[estado])  # Exploração: escolhe a melhor ação com base na tabela Q

        # Execução da ação no ambiente
        proximo_estado, recompensa, done, _, _ = env.step(acao)

        # Atualização da tabela Q
        q_antigo = q_table[estado, acao]
        proximo_maximo = np.max(q_table[proximo_estado])
        q_novo = (1 - alpha) * q_antigo + alpha * (recompensa + gamma * proximo_maximo)
        q_table[estado, acao] = q_novo

        # Contabilização de penalidades
        if recompensa == -10:
            penalidades += 1

        estado = proximo_estado  # Atualiza o estado para o próximo estado

    # Exibição do progresso a cada 100 episódios
    if i % 100 == 0:
        clear_output(wait=True)
        print(f'Episódio: {i}')

end_time = time.time()
tempo = end_time - start_time
print(f"Tempo de treinamento: {tempo} segundos")

# Avaliação do agente treinado
total_penalidades = 0
episodios = 50
frames = []

for _ in range(episodios):
    estado, _ = env.reset()
    penalidades, recompensa = 0, 0
    done = False

    while not done:
        acao = np.argmax(q_table[estado])  # Seleção da ação baseada na tabela Q
        prox_estado, recompensa, done, info, _ = env.step(acao)

        # Contabilização de penalidades
        if recompensa == -10:
            penalidades += 1

        # Armazenamento dos frames para exibição posterior
        frames.append({
            'frame': env.render(),
            'state': prox_estado,
            'action': acao,
            'reward': recompensa
        })

        estado = prox_estado  # Atualiza o estado para o próximo estado

    total_penalidades += penalidades

# Exibição dos resultados
print('Episódios:', episodios)
print('Penalidades:', total_penalidades)

# Exibição interativa dos frames de cada episódio
for frame in frames:
    clear_output(wait=True)
    print(frame['frame'])
    print('Estado', frame['state'])
    print('Ação', frame['action'])
    print('Recompensa', frame['reward'])
    time.sleep(.5)
