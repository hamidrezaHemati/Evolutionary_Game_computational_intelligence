import copy
from player import Player
import numpy as np
from config import CONFIG
import random
import math


class Evolution():

    def __init__(self, mode):
        self.mode = mode
        self.generation = 0

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def mutate(self, child):
        mutate_threshold = np.random.uniform(low=0, high=1, size=4)
        if mutate_threshold[0] < 0.3:
            child.nn.w0 += np.random.normal(size=child.nn.w0.shape)
        if mutate_threshold[1] < 0.3:
            child.nn.w1 += np.random.normal(size=child.nn.w1.shape)
        if mutate_threshold[2] < 0.3:
            child.nn.b0 += np.random.normal(size=child.nn.b0.shape)
        if mutate_threshold[3] < 0.3:
            child.nn.b1 += np.random.normal(size=child.nn.b1.shape)

    def wheel(self, prev_players):
        sigma_fitness = 0
        for player in prev_players:
            sigma_fitness += player.fitness
        player_probability = []
        cumulative_p = 0
        for i, player in enumerate(prev_players):
            cumulative_p += player.fitness / sigma_fitness
            player_probability.append([i, cumulative_p, player])
        return player_probability

    def roulette_wheel(self, num_players, player_probability):
        parents = []
        for i in range(num_players):
            uniform_random_number = random.uniform(0, 1)
            for player in player_probability:
                if player[1] > uniform_random_number:
                    parents.append(player[2])
                    break
        return parents

    def Q_tournament(self, num_players, prev_players, q):
        parents = []
        for i in range(num_players):
            uniform_random_number = np.random.uniform(low=0, high=len(prev_players), size=q)
            uniform_random_number = [math.floor(x) for x in uniform_random_number]
            selected_players = []
            for r in uniform_random_number:
                selected_players.append(prev_players[r])
            selected_players = sorted(selected_players, key=lambda player: player.fitness, reverse=True)
            # for k in selected_players:
            #     print(k.fitness, end=" ")
            # print()
            parents.append(selected_players[0])
        return parents

    def crossOver(self, parents):
        children = []
        for i in range(0, len(parents), 2):
            child1 = Player(mode=self.mode)
            child2 = Player(mode=self.mode)

            size_child_w0 = int(child1.nn.w0.shape[0] / 2)
            size_child_w1 = int(child1.nn.w1.shape[0] / 2)
            size_child_b0 = int(child1.nn.b0.shape[0] / 2)
            size_child_b1 = int(child1.nn.b1.shape[0] / 2)

            child1.nn.w0[0:size_child_w0, :] = parents[i].nn.w0[0:size_child_w0, :]
            child2.nn.w0[0:size_child_w0, :] = parents[i + 1].nn.w0[0:size_child_w0, :]
            child1.nn.w0[size_child_w0:, :] = parents[i + 1].nn.w0[size_child_w0:, :]
            child2.nn.w0[size_child_w0:, :] = parents[i].nn.w0[size_child_w0:, :]

            child1.nn.w1[0:size_child_w1, :] = parents[i].nn.w1[0:size_child_w1, :]
            child2.nn.w1[0:size_child_w1, :] = parents[i + 1].nn.w1[0:size_child_w1, :]
            child1.nn.w1[size_child_w1:, :] = parents[i + 1].nn.w1[size_child_w1:, :]
            child2.nn.w1[size_child_w1:, :] = parents[i].nn.w1[size_child_w1:, :]

            child1.nn.b0[0:size_child_b0, :] = parents[i].nn.b0[0:size_child_b0, :]
            child2.nn.b0[0:size_child_b0, :] = parents[i + 1].nn.b0[0:size_child_b0, :]
            child1.nn.b0[size_child_b0:, :] = parents[i + 1].nn.b0[size_child_b0:, :]
            child2.nn.b0[size_child_b0:, :] = parents[i].nn.b0[size_child_b0:, :]

            child1.nn.b1[0:size_child_b1, :] = parents[i].nn.b1[0:size_child_b1, :]
            child2.nn.b1[0:size_child_b1, :] = parents[i + 1].nn.b1[0:size_child_b1, :]
            child1.nn.b1[size_child_b1:, :] = parents[i + 1].nn.b1[size_child_b1:, :]
            child2.nn.b1[size_child_b1:, :] = parents[i].nn.b1[size_child_b1:, :]
            children.append(child1)
            children.append(child2)

        return children

    def generate_new_population(self, num_players, prev_players=None):

        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]
        else:

            # roulette wheel
            # players_probability = self.wheel(prev_players)
            # parents = self.roulette_wheel(num_players, players_probability)

            # Q-tournament
            parents = self.Q_tournament(num_players, prev_players, 2)

            children = []
            for parent in parents:
                children.append(copy.deepcopy(parent))

            # children = self.crossOver(parents)

            for child in children:
                self.mutate(child)

            return children

    def next_population_selection(self, players, num_players):
        # num_players example: 100
        # players: an array of `Player` objects
        fitnesses = []
        for player in players:
            fitnesses.append(player.fitness)

        # sort base
        selected = sorted(players, key=lambda player: player.fitness, reverse=True)  # sort by the fitness value
        # RW
        # players_probability = self.wheel(players)
        # selected = self.roulette_wheel(num_players, players_probability_probability)

        # plotting
        min_fitness = min(fitnesses)
        max_fitness = max(fitnesses)
        average_fitness = sum(fitnesses) / len(fitnesses)
        with open('fitnesses.txt', 'a') as filehandle:
            filehandle.write(f"{min_fitness} {average_fitness} {max_fitness}")
            filehandle.write("\n")

        return selected[: num_players]
