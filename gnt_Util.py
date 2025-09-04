import csv
import random
import sys
import numpy as np
# import tensorflow as tf
# from tensorflow.keras.layers import Dropout, Dense, GRU
# import matplotlib.pyplot as plt
# import os
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import math
import copy
import heapq



class text():

    def __init__(self):
        self._dc_cpu_resourse = 120
        self._dc_mem_resourse = 120
        self.max_bandwidth = 100
        self.dc_cost = 1000
        self.exceed_vnfs_dict = {}
        self.dcs_exceed_list = []
        self.dcs_dict = self.create_dc_topology(dc_number = 50)
        self.dcs_link_matrix = []
        self.installed_id = 0
        self.SFCs = {}
        self.SFCs_dc_path = {}
        self.bono_cpu = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.bono_mem = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    def return_sfcs(self):
        return self.SFCs

    def return_sfc_dc_path(self):
        return self.SFCs_dc_path

    def return_dcs_dict(self):
        # 返回dcs_dict字典
        return self.dcs_dict

    def return_dcs_link_matrix(self):
        return self.dcs_link_matrix

    def return_choose_vnfs(self):
        return self.exceed_vnfs_dict

    def return_dcs_exceed_list(self):
        return self.dcs_exceed_list

    def add_DC(self, dc_name, dc_cpuresource, dc_memresource):
        self.dcs_dict[dc_name]['dc_cpu_resource'] = dc_cpuresource
        self.dcs_dict[dc_name]['dc_mem_resource'] = dc_memresource
        return dc_name



    def delete_VNF(self, dc_name, vnf_name, vnf_cpu_resource, vnf_mem_resource):
        if vnf_name in self.dcs_dict[dc_name]['vnfs']:
            self.dcs_dict[dc_name]['dc_cpu_resource'] += vnf_cpu_resource
            self.dcs_dict[dc_name]['dc_mem_resource'] += vnf_mem_resource
            del self.dcs_dict[dc_name]['vnfs'][vnf_name]
        else:
            return 0

    def create_dc_topology(self, dc_number):
        dcs_dict = {}
        for i in range(0, dc_number):
            dcs_dict["dc"+str(i)] = {'dc_cpu_resource': self._dc_cpu_resourse, 'dc_mem_resource': self._dc_mem_resourse, 'vnfs': {}}
        return dcs_dict

    def dijkstra(self, graph, start, end, bandwidth):
        temp_graph = [[0] * len(graph[0]) for _ in range(len(graph))]
        for i in range(len(graph)):
            for j in range(len(graph[0])):
                if graph[i][j] > 0:
                    temp_graph[i][j] = self.max_bandwidth - graph[i][j]
                else:
                    temp_graph[i][j] = -1

        num_vertices = len(temp_graph)
        distances = [float('inf')] * num_vertices
        distances[start] = 0
        path = [None] * num_vertices
        visited = [False] * num_vertices  # 用于记录每个顶点是否已经被访问过
        priority_queue = [(0, start)]

        while priority_queue:
            current_distance, current_vertex = heapq.heappop(priority_queue)

            # 如果当前顶点已经被访问过，跳过
            if visited[current_vertex]:
                continue
            visited[current_vertex] = True

            for neighbor, weight in enumerate(temp_graph[current_vertex]):
                if weight == -1 or self.max_bandwidth - weight < bandwidth:
                    continue
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    path[neighbor] = current_vertex
                    heapq.heappush(priority_queue, (distance, neighbor))

        shortest_path = []
        current_vertex = end
        while current_vertex is not None:
            shortest_path.append(current_vertex)
            current_vertex = path[current_vertex]
        shortest_path.reverse()


        for i in range(0, len(shortest_path) - 1):
            self.dcs_link_matrix[shortest_path[i]][shortest_path[i + 1]] = self.dcs_link_matrix[shortest_path[i]][shortest_path[i + 1]] - bandwidth
            self.dcs_link_matrix[shortest_path[i + 1]][shortest_path[i]] = self.dcs_link_matrix[shortest_path[i + 1]][shortest_path[i]] - bandwidth

        return shortest_path



    def return_dijkstra_path(self, graph, start, end, bandwidth):
        temp_graph = [[0] * len(graph[0]) for _ in range(len(graph))]
        for i in range(len(graph)):
            for j in range(len(graph[0])):
                if graph[i][j] > 0:
                    temp_graph[i][j] = self.max_bandwidth - graph[i][j]
                else:
                    temp_graph[i][j] = -1

        num_vertices = len(temp_graph)
        distances = [float('inf')] * num_vertices
        distances[start] = 0
        path = [None] * num_vertices
        visited = [False] * num_vertices  # 用于记录每个顶点是否已经被访问过
        priority_queue = [(0, start)]

        while priority_queue:
            current_distance, current_vertex = heapq.heappop(priority_queue)

            # 如果当前顶点已经被访问过，跳过
            if visited[current_vertex]:
                continue
            visited[current_vertex] = True

            for neighbor, weight in enumerate(temp_graph[current_vertex]):
                if weight == -1 or self.max_bandwidth - weight < bandwidth:
                    continue
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    path[neighbor] = current_vertex
                    heapq.heappush(priority_queue, (distance, neighbor))

        shortest_path = []
        current_vertex = end
        while current_vertex is not None:
            shortest_path.append(current_vertex)
            current_vertex = path[current_vertex]
        shortest_path.reverse()
        return shortest_path

    def add_SFC(self, vnfs_list, **params):
        # vnfs_list:[[dc_name, vnf_name, cpu, mem],[]...]
        sfc_dc = []
        number_sfc_dc = []
        band_value = []
        for value in params.values():
            band_value.append(value)
        # print(band_value)
        if not self.dc_sfc_resource_check(vnfs_list=vnfs_list):
            return 0
        for item_list in vnfs_list:
            self.add_VNF(item_list[0], item_list[1], item_list[2], item_list[3])
        for sfc in vnfs_list:
            sfc_dc.append(sfc[0])
        for number in sfc_dc:
            number_sfc_dc.append(int(number.split("c")[-1]))
        # print(number_sfc_dc)
        all_dc_path = []
        for i in range(0, len(number_sfc_dc)-1):
            path = self.dijkstra(self.dcs_link_matrix, int(number_sfc_dc[i]), int(number_sfc_dc[i + 1]), int(band_value[0]))
            all_dc_path.append(path)
        new_path_list = []
        for i in range(len(all_dc_path)):
            if i == 0:
                new_path_list.append(all_dc_path[i])
            else:
                if all_dc_path[i][0] == new_path_list[-1][-1]:
                    new_path_list[-1].extend(all_dc_path[i][1:])
                else:
                    new_path_list.append(all_dc_path[i])
        tem_dict = {"sfc_vnf":vnfs_list}
        tem_dict.update(params)
        self.installed_id += 1
        self.SFCs[self.installed_id] = tem_dict
        self.SFCs_dc_path[self.installed_id] = new_path_list[0]
        return self.installed_id

    def add_VNF(self, dc_name, vnf_name, vnf_cpu_resource, vnf_mem_resource):
        # update
        if vnf_name not in self.dcs_dict[dc_name]['vnfs']:
            self.dcs_dict[dc_name]['vnfs'][vnf_name] = {'vnf_cpu_resource': 0, 'vnf_mem_resource': 0}
        if self.dcs_dict[dc_name]['dc_cpu_resource'] - vnf_cpu_resource > 0 and self.dcs_dict[dc_name]['dc_mem_resource'] - vnf_mem_resource > 0:
            self.dcs_dict[dc_name]['vnfs'][vnf_name]['vnf_cpu_resource'] = vnf_cpu_resource
            self.dcs_dict[dc_name]['vnfs'][vnf_name]['vnf_mem_resource'] = vnf_mem_resource
            self.dcs_dict[dc_name]['dc_cpu_resource'] -= vnf_cpu_resource
            self.dcs_dict[dc_name]['dc_mem_resource'] -= vnf_mem_resource
        else:
            return str(dc_name) + "中" + str(vnf_name) + "部署失败"
        return 1

    def GNT_VNF_choose(self, dc_name):
        # 这里计算dc_name物理节点中经过vnf的所有带宽
        band = 0
        number = 0
        dc_vnf_bandwidth_dict = {}
        dc_vnf_number_dict = {}
        dc_vnf_gnt_dict = {}
        for dc_vnf in self.dcs_dict[dc_name]["vnfs"].keys():
            for sfc in self.SFCs.keys():
                for i in self.SFCs[sfc]["sfc_vnf"]:
                    if(i[1] == dc_vnf):
                        number = number + 1
                        band = self.SFCs[sfc]["bandwidth"] + band
            dc_vnf_bandwidth_dict[dc_vnf] = band
            dc_vnf_number_dict[dc_vnf] = number
        for key in dc_vnf_bandwidth_dict.keys():
            dc_vnf_gnt_dict[key] = dc_vnf_bandwidth_dict[key]
        return dc_vnf_gnt_dict

    def fitness_migration(self, position, dc_name, bandwidth):
        temp_exceed_vnfs_dict = copy.deepcopy(self.exceed_vnfs_dict)
        temp_dcs_exceed_list = copy.deepcopy(self.dcs_exceed_list)
        temp_dcs_dict = copy.deepcopy(self.dcs_dict)
        temp_dcs_link_matrix = copy.deepcopy(self.dcs_link_matrix)
        temp_installed_id = copy.deepcopy(self.installed_id)
        temp_SFCs = copy.deepcopy(self.SFCs)


        # 处理节点
        temp_dicts = {}
        vnf_gnt_dict = self.GNT_VNF_choose(dc_name)
        result_min = min(vnf_gnt_dict, key=lambda x: vnf_gnt_dict[x])
        # print(result_min)
        cpu_resource = self.dcs_dict[dc_name]['vnfs'][result_min]['vnf_cpu_resource']
        mem_resource = self.dcs_dict[dc_name]['vnfs'][result_min]['vnf_mem_resource']
        self.dcs_dict['dc' + str(position)]['dc_cpu_resource'] = self.dcs_dict['dc' + str(position)]['dc_cpu_resource'] - cpu_resource
        self.dcs_dict['dc' + str(position)]['dc_mem_resource'] = self.dcs_dict['dc' + str(position)]['dc_mem_resource'] - mem_resource
        temp_dicts[result_min] = self.dcs_dict[dc_name]['vnfs'].get(result_min)
        self.dcs_dict['dc' + str(position)]['vnfs'].update(temp_dicts)
        self.dcs_dict[dc_name]['dc_cpu_resource'] = self.dcs_dict[dc_name]['dc_cpu_resource'] + cpu_resource
        self.dcs_dict[dc_name]['dc_mem_resource'] = self.dcs_dict[dc_name]['dc_mem_resource'] + mem_resource
        del self.dcs_dict[dc_name]['vnfs'][result_min]

        # 处理链路
        # 首先要找到所有sfc中包含需要迁移的vnf的的sfc
        sfc_name_migration = []
        for sfc_name in self.SFCs.keys():
            for sfc_link in self.SFCs[sfc_name]['sfc_vnf']:
                if(sfc_link[1] == result_min):
                    sfc_name_migration.append(sfc_name)
        # print(sfc_name_migration)

        sfcs_link = []
        sfc_temp_link = []
        for i in sfc_name_migration:
            for sfc_temp in self.SFCs[i]['sfc_vnf']:
                sfc_temp_link.append(sfc_temp[0])
            sfcs_link.append(sfc_temp_link)
            sfc_temp_link = []
        # print(sfcs_link)
        # 此时sfcs_link列表里存放了所有包含需要迁移vnf的sfc存放的物理节点

        for sfc in sfcs_link:
            # print(sfc)
            for i in range(0, len(sfc)):
                if(sfc[i] == dc_name):
                    temp = i
                    # print(sfc)
                    if(temp==0):
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, int(sfc[i].split('dc')[-1]), int(sfc[i+1].split('dc')[-1]), bandwidth)
                        for j in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[j]][shortest_path[j + 1]] = self.dcs_link_matrix[shortest_path[j]][shortest_path[j + 1]] + vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[j + 1]][shortest_path[j]] = self.dcs_link_matrix[shortest_path[j + 1]][shortest_path[j]] + vnf_gnt_dict[result_min]
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, position, int(sfc[i+1].split('dc')[-1]), bandwidth)
                        # print(shortest_path)
                        for k in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[k]][shortest_path[k + 1]] = self.dcs_link_matrix[shortest_path[k]][shortest_path[k + 1]] - vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[k + 1]][shortest_path[k]] = self.dcs_link_matrix[shortest_path[k + 1]][shortest_path[k]] - vnf_gnt_dict[result_min]
                    if(temp==len(sfc)-1):
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, int(sfc[i-1].split('dc')[-1]), int(sfc[i].split('dc')[-1]), bandwidth)
                        # print(shortest_path)
                        for j in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[j]][shortest_path[j + 1]] = self.dcs_link_matrix[shortest_path[j]][shortest_path[j + 1]] + vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[j + 1]][shortest_path[j]] = self.dcs_link_matrix[shortest_path[j + 1]][shortest_path[j]] + vnf_gnt_dict[result_min]
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix,  position, int(sfc[i - 1].split('dc')[-1]), bandwidth)
                        # print(shortest_path)
                        for k in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[k]][shortest_path[k + 1]] = self.dcs_link_matrix[shortest_path[k]][shortest_path[k + 1]] - vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[k + 1]][shortest_path[k]] = self.dcs_link_matrix[shortest_path[k + 1]][shortest_path[k]] - vnf_gnt_dict[result_min]
                    if(temp!=len(sfc)-1 and temp!=0):
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, int(sfc[i].split('dc')[-1]),int(sfc[i + 1].split('dc')[-1]), bandwidth)
                        for j in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[j]][shortest_path[j + 1]] = self.dcs_link_matrix[shortest_path[j]][shortest_path[j + 1]] + vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[j + 1]][shortest_path[j]] = self.dcs_link_matrix[shortest_path[j + 1]][shortest_path[j]] + vnf_gnt_dict[result_min]
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, position, int(sfc[i + 1].split('dc')[-1]), bandwidth)
                        for k in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[k]][shortest_path[k + 1]] = self.dcs_link_matrix[shortest_path[k]][shortest_path[k + 1]] - vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[k + 1]][shortest_path[k]] = self.dcs_link_matrix[shortest_path[k + 1]][shortest_path[k]] - vnf_gnt_dict[result_min]

                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, int(sfc[i - 1].split('dc')[-1]),int(sfc[i].split('dc')[-1]), bandwidth)
                        for l in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[l]][shortest_path[l + 1]] = self.dcs_link_matrix[shortest_path[l]][shortest_path[l + 1]] + vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[l + 1]][shortest_path[l]] = self.dcs_link_matrix[shortest_path[l + 1]][shortest_path[l]] + vnf_gnt_dict[result_min]
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, position, int(sfc[i - 1].split('dc')[-1]), bandwidth)
                        for m in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[m]][shortest_path[m + 1]] = self.dcs_link_matrix[shortest_path[m]][shortest_path[m + 1]] - vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[m + 1]][shortest_path[m]] = self.dcs_link_matrix[shortest_path[m + 1]][shortest_path[m]] - vnf_gnt_dict[result_min]

        # variance_cpu_resource, variance_mem_resource = self.return_dcs_resource_variance()
        # fitness = 0.5 * variance_cpu_resource + 0.5 * variance_mem_resource

        fitness = 1/self.return_migration_costs(position, dc_name)

        self.exceed_vnfs_dict = temp_exceed_vnfs_dict
        self.dcs_exceed_list = temp_dcs_exceed_list
        self.dcs_dict = temp_dcs_dict
        self.dcs_link_matrix = temp_dcs_link_matrix
        self.installed_id = temp_installed_id
        self.SFCs = temp_SFCs


        return fitness

    def migration(self, position, dc_name, bandwidth):
        temp_dicts = {}
        vnf_gnt_dict = self.GNT_VNF_choose(dc_name)
        result_min = min(vnf_gnt_dict, key=lambda x: vnf_gnt_dict[x])
        cpu_resource = self.dcs_dict[dc_name]['vnfs'][result_min]['vnf_cpu_resource']
        mem_resource = self.dcs_dict[dc_name]['vnfs'][result_min]['vnf_mem_resource']
        self.dcs_dict['dc' + str(position)]['dc_cpu_resource'] = self.dcs_dict['dc' + str(position)][
                                                                     'dc_cpu_resource'] - cpu_resource
        self.dcs_dict['dc' + str(position)]['dc_mem_resource'] = self.dcs_dict['dc' + str(position)][
                                                                     'dc_mem_resource'] - mem_resource
        temp_dicts[result_min] = self.dcs_dict[dc_name]['vnfs'].get(result_min)
        self.dcs_dict['dc' + str(position)]['vnfs'].update(temp_dicts)
        self.dcs_dict[dc_name]['dc_cpu_resource'] = self.dcs_dict[dc_name]['dc_cpu_resource'] + cpu_resource
        self.dcs_dict[dc_name]['dc_mem_resource'] = self.dcs_dict[dc_name]['dc_mem_resource'] + mem_resource
        del self.dcs_dict[dc_name]['vnfs'][result_min]

        # 处理链路
        # 首先要找到所有sfc中包含需要迁移的vnf的的sfc
        sfc_name_migration = []
        for sfc_name in self.SFCs.keys():
            for sfc_link in self.SFCs[sfc_name]['sfc_vnf']:
                if (sfc_link[1] == result_min):
                    sfc_name_migration.append(sfc_name)

        sfcs_link = []
        sfc_temp_link = []
        for i in sfc_name_migration:
            for sfc_temp in self.SFCs[i]['sfc_vnf']:
                sfc_temp_link.append(sfc_temp[0])
            sfcs_link.append(sfc_temp_link)
            sfc_temp_link = []
        # 此时sfcs_link列表里存放了所有包含需要迁移vnf的sfc存放的物理节点

        for sfc in sfcs_link:
            # print(sfc)
            for i in range(0, len(sfc)):
                if (sfc[i] == dc_name):
                    temp = i
                    # print(sfc)
                    if (temp == 0):
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, int(sfc[i].split('dc')[-1], bandwidth),
                                                                  int(sfc[i + 1].split('dc')[-1]))
                        for j in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[j]][shortest_path[j + 1]] = \
                            self.dcs_link_matrix[shortest_path[j]][shortest_path[j + 1]] + vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[j + 1]][shortest_path[j]] = \
                            self.dcs_link_matrix[shortest_path[j + 1]][shortest_path[j]] + vnf_gnt_dict[result_min]
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, position,
                                                                  int(sfc[i + 1].split('dc')[-1]), bandwidth)
                        # print(shortest_path)
                        for k in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[k]][shortest_path[k + 1]] = \
                            self.dcs_link_matrix[shortest_path[k]][shortest_path[k + 1]] - vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[k + 1]][shortest_path[k]] = \
                            self.dcs_link_matrix[shortest_path[k + 1]][shortest_path[k]] - vnf_gnt_dict[result_min]
                    if (temp == len(sfc) - 1):
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, int(sfc[i - 1].split('dc')[-1]),
                                                                  int(sfc[i].split('dc')[-1]), bandwidth)
                        # print(shortest_path)
                        for j in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[j]][shortest_path[j + 1]] = \
                            self.dcs_link_matrix[shortest_path[j]][shortest_path[j + 1]] + vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[j + 1]][shortest_path[j]] = \
                            self.dcs_link_matrix[shortest_path[j + 1]][shortest_path[j]] + vnf_gnt_dict[result_min]
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, position,
                                                                  int(sfc[i - 1].split('dc')[-1]), bandwidth)

                        for k in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[k]][shortest_path[k + 1]] = \
                            self.dcs_link_matrix[shortest_path[k]][shortest_path[k + 1]] - vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[k + 1]][shortest_path[k]] = \
                            self.dcs_link_matrix[shortest_path[k + 1]][shortest_path[k]] - vnf_gnt_dict[result_min]
                    if (temp != len(sfc) - 1 and temp != 0):
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, int(sfc[i].split('dc')[-1]),
                                                                  int(sfc[i + 1].split('dc')[-1]), bandwidth)
                        for j in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[j]][shortest_path[j + 1]] = \
                            self.dcs_link_matrix[shortest_path[j]][shortest_path[j + 1]] + vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[j + 1]][shortest_path[j]] = \
                            self.dcs_link_matrix[shortest_path[j + 1]][shortest_path[j]] + vnf_gnt_dict[result_min]
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, position,
                                                                  int(sfc[i + 1].split('dc')[-1]), bandwidth)
                        for k in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[k]][shortest_path[k + 1]] = \
                            self.dcs_link_matrix[shortest_path[k]][shortest_path[k + 1]] - vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[k + 1]][shortest_path[k]] = \
                            self.dcs_link_matrix[shortest_path[k + 1]][shortest_path[k]] - vnf_gnt_dict[result_min]

                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, int(sfc[i - 1].split('dc')[-1]),
                                                                  int(sfc[i].split('dc')[-1]), bandwidth)
                        for l in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[l]][shortest_path[l + 1]] = \
                            self.dcs_link_matrix[shortest_path[l]][shortest_path[l + 1]] + vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[l + 1]][shortest_path[l]] = \
                            self.dcs_link_matrix[shortest_path[l + 1]][shortest_path[l]] + vnf_gnt_dict[result_min]
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, position,
                                                                  int(sfc[i - 1].split('dc')[-1]), bandwidth)
                        for m in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[m]][shortest_path[m + 1]] = \
                            self.dcs_link_matrix[shortest_path[m]][shortest_path[m + 1]] - vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[m + 1]][shortest_path[m]] = \
                            self.dcs_link_matrix[shortest_path[m + 1]][shortest_path[m]] - vnf_gnt_dict[result_min]

    def dc_sfc_resource_check(self, vnfs_list):
        # vnfs_list:[[dc_name, vnf_name, cpu, mem],[]...]
        # check
        dc_resourse_temp = {}
        for item_list in vnfs_list:
            dc_name = item_list[0]
            vnf_name = item_list[1]
            vnf_cpu_resource = item_list[2]
            vnf_mem_resource = item_list[3]
            if dc_name not in dc_resourse_temp:
                dc_resourse_temp[dc_name] = {'dc_cpu_resource': 0, 'dc_mem_resource': 0}
                dc_resourse_temp[dc_name]['dc_cpu_resource'] = self.dcs_dict[dc_name]['dc_cpu_resource']
                dc_resourse_temp[dc_name]['dc_mem_resource'] = self.dcs_dict[dc_name]['dc_mem_resource']
            dc_resourse_temp[dc_name]['dc_cpu_resource'] -= vnf_cpu_resource
            dc_resourse_temp[dc_name]['dc_mem_resource'] -= vnf_mem_resource
            if dc_resourse_temp[dc_name]['dc_cpu_resource'] < 0 or dc_resourse_temp[dc_name]['dc_mem_resource'] < 0:
                print("run out of resourse:{}", item_list)
                return 0
        return 1

    def check_dc_threshold_exceed(self, dc_cpu_resource_percent, dc_mem_resource_percent):
        '''
        检查超出阈值的物理节点
        :param dc_cpu_resource_percent:
        :param dc_mem_resource_percent:
        :return:
        '''
        dc_threshold_exceed_list = []
        for key in self.dcs_dict.keys():
            if dc_cpu_resource_percent*self.dcs_dict[key]['dc_cpu_resource'] + dc_mem_resource_percent*self.dcs_dict[key]['dc_mem_resource'] < 40:
                dc_threshold_exceed_list.append(key)
                # print("物理节点"+key+"过载！！！！！！")

        self.dcs_exceed_list = dc_threshold_exceed_list
        return dc_threshold_exceed_list


    def return_dcs_resource_ave(self):
        sum_cpu_resource = 0
        sum_mem_resource = 0
        for key in self.dcs_dict.keys():
            sum_cpu_resource = self.dcs_dict[key]['dc_cpu_resource'] + sum_cpu_resource
            sum_mem_resource = self.dcs_dict[key]['dc_mem_resource'] + sum_mem_resource
        ave_cpu_resource = sum_cpu_resource/len(self.dcs_dict)
        ave_mem_resource = sum_mem_resource / len(self.dcs_dict)
        return ave_cpu_resource, ave_mem_resource

    def return_dcs_resource_variance(self):
        ave_cpu_resource, ave_mem_resource = self.return_dcs_resource_ave()
        variance_cpu_resource = 0
        variance_mem_resource = 0
        for key in self.dcs_dict.keys():
            variance_cpu_resource = (self.dcs_dict[key]['dc_cpu_resource'] - ave_cpu_resource) * (self.dcs_dict[key]['dc_cpu_resource'] - ave_cpu_resource) + variance_cpu_resource
            variance_mem_resource = (self.dcs_dict[key]['dc_mem_resource'] - ave_mem_resource) * (self.dcs_dict[key]['dc_mem_resource'] - ave_mem_resource) + variance_mem_resource
        return len(self.dcs_dict)/variance_cpu_resource, len(self.dcs_dict)/variance_mem_resource

    def return_migration_costs(self, position, dc_name):
        temp = self.return_dijkstra_path(self.dcs_link_matrix, position, int(dc_name.split("dc")[-1]))
        vnf_gnt_dict = self.GNT_VNF_choose(dc_name)
        result_min = min(vnf_gnt_dict, key=lambda x: vnf_gnt_dict[x])
        value = 0.5 * self.dcs_dict["dc" + str(position)]['dc_cpu_resource'] + 0.5 * self.dcs_dict["dc" + str(position)]['dc_mem_resource']
        return len(temp) * (value/120 + vnf_gnt_dict[result_min]/100)



    def clear_exceed_vnfs_dict(self):
        self.exceed_vnfs_dict.clear()

    # def creaate_dc_link(self):
    #     self.dcs_link_matrix = [
    #         [-1, 100, -1, -1, -1, -1, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    #         [100, -1, -1, -1, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    #         [-1, -1, -1, -1, 100, -1, -1, 100, 100, -1, -1, -1, -1, -1, -1, -1],
    #         [-1, -1, -1, -1, -1, 100, 100, -1, -1, -1, -1, 100, -1, -1, -1, -1],
    #         [-1, 100, 100, -1, -1, -1, 100, -1, -1, 100, -1, -1, -1, -1, -1, -1],
    #         [-1, -1, -1, 100, -1, -1, -1, -1, -1, -1, -1, 100, -1, -1, -1, -1],
    #         [100, -1, -1, 100, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    #         [-1, -1, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 100, -1, -1],
    #         [-1, -1, 100, -1, -1, -1, -1, -1, -1, 100, -1, 100, -1, 100, -1, -1],
    #         [-1, -1, -1, -1, 100, -1, -1, -1, 100, -1, 100, -1, -1, -1, -1, 100],
    #         [-1, -1, -1, -1, -1, -1, -1, -1, -1, 100, -1, -1, 100, -1, 100, -1],
    #         [-1, -1, -1, 100, -1, 100, -1, -1, 100, -1, -1, -1, -1, -1, -1, -1],
    #         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 100, -1, -1, -1, 100, -1],
    #         [-1, -1, -1, -1, -1, -1, -1, 100, 100, -1, -1, -1, -1, -1, -1, 100],
    #         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 100, -1, 100, -1, -1, -1],
    #         [-1, -1, -1, -1, -1, -1, -1, -1, -1, 100, -1, -1, -1, 100, -1, -1]
    #     ]

    # def creaate_dc_link(self):
    #     self.dcs_link_matrix = [
    #         [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    #         [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    #         [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    #         [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    #         [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    #         [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    #         [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    #         [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    #         [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    #         [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    #         [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    #         [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    #         [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    #         [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    #         [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    #         [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    #     ]
    #
    # def create_dc_link(self):
    #     self.dcs_link_matrix = [
    #         [100, 100, -1, 100, -1, 100, -1, -1, 100, -1, -1, -1, -1, -1, 100, -1, 100, 100, 100, 100, -1, -1, -1, -1, 100, -1, 100, -1, -1, 100, 100, -1, -1, 100, 100, -1, -1, 100, -1, 100, 100, -1, 100, 100, 100, -1, -1, -1, 100, -1],
    #         [100, 100, -1, -1, -1, -1, 100, 100, 100, -1, 100, 100, 100, 100, -1, -1, 100, -1, 100, 100, -1, 100, -1, 100, -1, 100, 100, 100, 100, -1, 100, 100, -1, 100, 100, 100, 100, -1, 100, -1, 100, 100, 100, -1, -1, 100, -1, -1, -1, 100],
    #         [100, -1, -1, 100, -1, 100, -1, 100, -1, -1, 100, 100, 100, -1, -1, 100, -1, -1, -1, -1, -1, 100, -1, 100, -1, 100, -1, 100, 100, -1, -1, 100, 100, -1, -1, -1, -1, 100, 100, 100, 100, 100, 100, -1, -1, 100, 100, 100, 100, 100],
    #         [-1, 100, -1, -1, -1, 100, -1, 100, 100, 100, -1, 100, -1, -1, 100, -1, -1, 100, 100, 100, 100, 100, -1, -1, 100, -1, 100, -1, -1, -1, -1, 100, -1, 100, 100, -1, -1, -1, -1, -1, -1, -1, -1, 100, 100, -1, -1, -1, 100, -1],
    #         [100, -1, -1, -1, -1, -1, -1, 100, -1, -1, 100, -1, 100, 100, -1, -1, -1, -1, -1, -1, -1, -1, 100, 100, -1, 100, 100, -1, -1, 100, -1, -1, 100, -1, 100, -1, 100, -1, 100, -1, -1, -1, -1, 100, -1, -1, -1, 100, 100, -1],
    #         [-1, 100, 100, 100, 100, 100, -1, 100, 100, -1, -1, 100, -1, 100, 100, 100, -1, 100, -1, -1, 100, -1, 100, 100, -1, 100, -1, 100, 100, -1, 100, 100, -1, 100, -1, 100, 100, -1, -1, 100, 100, -1, -1, 100, 100, -1, -1, -1, 100, -1],
    #         [-1, 100, -1, 100, -1, -1, 100, 100, -1, 100, 100, -1, 100, -1, -1, 100, 100, -1, 100, -1, 100, 100, 100, 100, 100, 100, 100, 100, -1, -1, 100, 100, -1, 100, -1, -1, 100, 100, -1, 100, -1, -1, -1, 100, -1, -1, 100, 100, 100, -1],
    #         [-1, 100, 100, 100, -1, 100, 100, 100, -1, -1, -1, -1, 100, -1, -1, -1, 100, 100, 100, 100, -1, 100, 100, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 100, 100, 100, -1, 100, 100, -1, -1, 100, -1, -1, -1, 100, 100, 100, -1],
    #         [-1, 100, -1, 100, -1, 100, 100, 100, -1, -1, -1, -1, -1, 100, -1, 100, -1, -1, 100, -1, 100, 100, -1, 100, -1, 100, 100, 100, -1, -1, 100, 100, 100, 100, 100, 100, -1, -1, -1, 100, -1, -1, 100, -1, 100, 100, 100, 100, 100, -1],
    #         [-1, 100, -1, 100, -1, 100, -1, 100, 100, -1, 100, 100, 100, -1, -1, -1, 100, 100, -1, 100, -1, -1, 100, -1, 100, 100, 100, -1, -1, 100, -1, 100, 100, 100, 100, -1, -1, -1, 100, -1, -1, 100, -1, -1, 100, -1, 100, -1, 100, 100],
    #         [100, 100, -1, 100, -1, 100, 100, 100, -1, 100, -1, -1, -1, 100, -1, 100, -1, 100, -1, 100, 100, -1, -1, -1, 100, -1, 100, 100, 100, -1, 100, -1, 100, -1, 100, -1, -1, 100, -1, 100, -1, -1, -1, -1, 100, -1, -1, 100, 100, 100],
    #         [-1, -1, -1, 100, -1, 100, -1, 100, -1, -1, -1, -1, 100, 100, -1, -1, 100, 100, -1, 100, -1, 100, -1, -1, -1, 100, -1, 100, -1, -1, -1, -1, -1, 100, 100, -1, 100, -1, 100, -1, 100, 100, 100, 100, -1, 100, 100, 100, 100, -1],
    #         [100, -1, -1, 100, 100, 100, -1, 100, -1, 100, -1, 100, 100, 100, 100, 100, 100, 100, -1, -1, -1, -1, 100, 100, -1, 100, -1, 100, -1, 100, -1, 100, -1, -1, 100, 100, 100, -1, 100, -1, 100, 100, 100, -1, -1, 100, -1, 100, 100, -1],
    #         [100, -1, -1, 100, -1, 100, 100, -1, 100, -1, 100, 100, -1, 100, 100, 100, -1, 100, -1, -1, 100, 100, 100, -1, 100, -1, -1, -1, -1, -1, 100, -1, 100, 100, 100, 100, 100, 100, -1, -1, -1, 100, 100, 100, -1, -1, -1, 100, -1, -1],
    #         [100, 100, 100, -1, 100, -1, 100, 100, -1, -1, 100, -1, 100, 100, -1, -1, 100, -1, 100, -1, 100, 100, 100, -1, -1, 100, 100, -1, -1, 100, -1, -1, 100, 100, 100, 100, -1, -1, 100, -1, 100, -1, -1, -1, 100, -1, 100, -1, 100, -1],
    #         [-1, 100, 100, 100, 100, -1, -1, -1, -1, 100, -1, -1, 100, -1, 100, 100, -1, 100, 100, -1, -1, 100, 100, 100, -1, 100, -1, -1, -1, 100, 100, -1, 100, -1, -1, -1, 100, 100, -1, -1, -1, -1, -1, 100, 100, -1, 100, 100, -1, 100],
    #         [100, -1, 100, -1, 100, -1, 100, 100, 100, -1, 100, -1, 100, -1, -1, -1, -1, -1, -1, 100, 100, 100, -1, -1, 100, 100, -1, -1, 100, -1, -1, -1, -1, 100, 100, 100, -1, 100, 100, 100, -1, -1, -1, 100, -1, 100, 100, -1, -1, 100],
    #         [-1, 100, 100, 100, 100, 100, -1, 100, 100, -1, 100, 100, -1, -1, 100, 100, 100, -1, -1, -1, -1, 100, 100, -1, -1, 100, 100, 100, -1, -1, -1, -1, -1, -1, 100, -1, -1, -1, -1, 100, 100, -1, -1, 100, 100, -1, -1, 100, -1, 100],
    #         [100, 100, -1, -1, -1, -1, 100, 100, -1, -1, -1, -1, 100, -1, 100, 100, 100, -1, 100, 100, -1, 100, -1, -1, -1, 100, 100, -1, 100, 100, 100, 100, 100, 100, -1, 100, 100, 100, 100, 100, -1, 100, 100, 100, -1, -1, 100, -1, -1, 100],
    #         [-1, -1, -1, -1, -1, -1, 100, 100, -1, 100, 100, 100, 100, -1, -1, -1, 100, -1, -1, 100, 100, -1, -1, 100, 100, -1, -1, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 100, 100, 100, -1, 100, 100, 100, -1, -1, -1, -1, -1],
    #         [100, -1, 100, -1, -1, 100, 100, -1, -1, 100, 100, -1, -1, 100, 100, 100, -1, 100, 100, -1, 100, -1, 100, 100, -1, -1, 100, -1, 100, -1, -1, 100, -1, 100, -1, -1, 100, 100, -1, 100, 100, -1, -1, 100, 100, -1, -1, 100, 100, -1],
    #         [100, 100, 100, -1, 100, 100, -1, -1, 100, -1, 100, -1, 100, 100, 100, 100, 100, 100, -1, -1, -1, 100, 100, 100, 100, 100, -1, 100, 100, -1, 100, 100, 100, 100, 100, -1, 100, -1, -1, -1, 100, -1, 100, 100, -1, -1, 100, -1, -1, 100],
    #         [100, 100, -1, 100, 100, -1, 100, -1, -1, -1, 100, 100, 100, -1, 100, -1, 100, -1, 100, 100, 100, 100, -1, -1, 100, -1, 100, -1, 100, 100, 100, 100, -1, -1, -1, -1, -1, -1, 100, 100, 100, 100, 100, 100, -1, 100, -1, -1, 100, 100],
    #         [-1, 100, -1, 100, -1, 100, -1, 100, -1, -1, -1, 100, 100, 100, -1, -1, 100, -1, -1, -1, -1, -1, 100, -1, 100, 100, 100, 100, 100, -1, 100, -1, -1, 100, -1, -1, -1, 100, 100, -1, 100, -1, -1, 100, 100, -1, 100, 100, 100, -1],
    #         [-1, -1, 100, -1, -1, 100, 100, 100, 100, 100, -1, -1, -1, -1, 100, -1, 100, -1, -1, -1, 100, -1, 100, -1, -1, 100, -1, -1, -1, -1, 100, -1, -1, -1, 100, 100, 100, -1, 100, 100, -1, 100, -1, 100, 100, 100, -1, 100, 100, -1],
    #         [-1, 100, -1, 100, 100, -1, 100, -1, -1, -1, 100, 100, -1, 100, -1, 100, 100, 100, -1, 100, -1, -1, -1, 100, 100, -1, 100, 100, -1, -1, 100, 100, 100, 100, -1, -1, -1, 100, 100, 100, -1, 100, 100, -1, 100, 100, -1, -1, -1, -1],
    #         [-1, -1, 100, -1, 100, 100, -1, -1, 100, 100, -1, 100, 100, 100, 100, -1, 100, 100, 100, 100, -1, 100, -1, -1, -1, -1, 100, 100, 100, 100, 100, -1, -1, -1, -1, -1, 100, -1, -1, -1, -1, -1, 100, -1, 100, 100, -1, 100, -1, -1],
    #         [-1, -1, 100, -1, 100, -1, 100, 100, -1, 100, 100, 100, 100, 100, 100, 100, -1, 100, -1, 100, -1, 100, 100, -1, 100, -1, -1, 100, -1, 100, 100, 100, 100, -1, -1, -1, 100, 100, -1, -1, -1, 100, 100, 100, -1, 100, -1, 100, -1, -1],
    #         [-1, -1, -1, 100, -1, -1, 100, 100, 100, 100, -1, -1, -1, 100, -1, 100, 100, 100, 100, -1, 100, -1, 100, 100, 100, -1, -1, 100, -1, 100, -1, 100, 100, 100, -1, -1, -1, 100, 100, 100, -1, -1, -1, -1, -1, 100, 100, 100, -1, -1],
    #         [-1, -1, 100, 100, 100, -1, 100, 100, -1, 100, -1, 100, 100, -1, -1, 100, 100, 100, -1, 100, 100, 100, 100, -1, 100, -1, 100, 100, 100, 100, 100, -1, 100, 100, -1, -1, 100, -1, -1, 100, 100, -1, -1, 100, -1, -1, 100, -1, 100, -1],
    #         [100, -1, -1, -1, 100, 100, -1, 100, -1, -1, -1, -1, 100, -1, 100, 100, 100, -1, 100, 100, -1, 100, 100, -1, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, -1, -1, -1, 100, -1, -1, 100, 100, 100, 100, 100, 100],
    #         [-1, 100, -1, 100, 100, -1, -1, -1, -1, 100, -1, -1, 100, 100, 100, 100, 100, 100, 100, -1, 100, 100, -1, -1, -1, -1, -1, 100, -1, 100, -1, -1, -1, -1, 100, -1, 100, 100, 100, -1, 100, -1, -1, 100, 100, 100, 100, 100, 100, -1],
    #         [100, 100, -1, 100, 100, -1, -1, -1, -1, 100, 100, -1, 100, -1, 100, 100, 100, -1, -1, 100, 100, 100, 100, -1, 100, 100, 100, -1, -1, -1, 100, -1, -1, -1, -1, 100, 100, -1, 100, -1, 100, -1, -1, 100, 100, -1, 100, -1, -1, 100],
    #         [-1, -1, -1, 100, -1, 100, -1, 100, 100, -1, -1, -1, 100, 100, 100, 100, 100, -1, 100, 100, 100, 100, -1, 100, -1, -1, 100, -1, 100, 100, 100, 100, 100, 100, -1, 100, -1, -1, -1, -1, -1, 100, -1, 100, -1, 100, 100, 100, -1, -1],
    #         [100, 100, -1, -1, 100, 100, 100, -1, -1, 100, 100, 100, -1, 100, 100, -1, -1, -1, -1, -1, -1, -1, 100, 100, 100, -1, 100, -1, 100, -1, -1, 100, 100, 100, -1, -1, 100, 100, 100, 100, -1, 100, -1, 100, -1, -1, -1, 100, -1, 100],
    #         [-1, 100, -1, -1, 100, 100, -1, -1, -1, 100, 100, -1, -1, 100, 100, 100, -1, -1, -1, 100, -1, 100, 100, 100, -1, -1, -1, -1, -1, 100, 100, 100, 100, -1, -1, -1, 100, 100, 100, -1, -1, 100, -1, 100, 100, -1, -1, -1, -1, -1],
    #         [100, -1, -1, -1, 100, 100, 100, -1, 100, 100, -1, 100, 100, 100, 100, -1, 100, -1, 100, 100, -1, -1, 100, 100, -1, 100, -1, 100, 100, 100, -1, -1, -1, -1, 100, -1, -1, 100, -1, -1, -1, -1, 100, 100, 100, -1, 100, -1, -1, -1],
    #         [100, 100, 100, 100, -1, -1, -1, 100, -1, 100, -1, -1, -1, -1, 100, 100, 100, 100, 100, 100, -1, 100, 100, 100, -1, -1, 100, 100, -1, 100, 100, 100, -1, 100, -1, 100, -1, -1, 100, -1, 100, -1, 100, 100, -1, 100, -1, 100, -1, -1],
    #         [100, 100, 100, -1, -1, -1, -1, -1, 100, 100, 100, -1, -1, -1, 100, -1, -1, -1, -1, 100, 100, -1, -1, 100, 100, 100, 100, 100, 100, -1, 100, 100, 100, 100, 100, 100, 100, 100, -1, 100, -1, -1, 100, 100, -1, 100, -1, -1, -1, 100],
    #         [100, -1, 100, -1, -1, -1, -1, 100, -1, 100, -1, 100, -1, -1, 100, 100, -1, -1, -1, -1, -1, -1, -1, 100, 100, 100, 100, 100, -1, 100, 100, -1, 100, 100, -1, -1, 100, -1, -1, -1, -1, 100, 100, 100, -1, 100, 100, -1, -1, -1],
    #         [100, 100, -1, -1, -1, -1, 100, -1, 100, 100, -1, 100, -1, -1, 100, 100, -1, 100, -1, 100, -1, -1, -1, 100, -1, 100, 100, -1, 100, 100, -1, -1, 100, -1, -1, -1, -1, 100, -1, -1, -1, 100, -1, 100, 100, 100, -1, -1, 100, -1],
    #         [100, -1, 100, -1, 100, 100, -1, 100, 100, -1, -1, -1, 100, 100, 100, -1, -1, -1, -1, -1, -1, 100, -1, 100, -1, 100, -1, -1, 100, -1, -1, 100, -1, 100, -1, -1, -1, -1, 100, -1, -1, 100, 100, 100, -1, -1, 100, -1, 100, -1],
    #         [100, 100, -1, 100, 100, -1, 100, -1, -1, -1, -1, 100, 100, -1, -1, -1, -1, 100, -1, 100, 100, -1, -1, 100, -1, 100, 100, -1, -1, -1, -1, 100, 100, 100, 100, 100, 100, 100, 100, 100, -1, 100, -1, -1, -1, 100, -1, -1, -1, -1],
    #         [100, -1, 100, 100, -1, -1, -1, -1, 100, -1, -1, -1, -1, -1, 100, -1, 100, -1, -1, 100, -1, -1, -1, 100, -1, 100, -1, -1, 100, -1, -1, -1, -1, -1, 100, -1, 100, -1, -1, -1, 100, 100, 100, 100, 100, 100, -1, -1, 100, -1],
    #         [100, -1, -1, -1, -1, -1, 100, 100, 100, 100, 100, -1, -1, 100, -1, 100, -1, 100, -1, -1, 100, -1, 100, -1, 100, -1, 100, -1, -1, -1, 100, 100, 100, -1, 100, -1, -1, -1, 100, 100, 100, 100, 100, 100, -1, 100, 100, 100, -1, -1],
    #         [100, 100, -1, -1, 100, 100, -1, -1, 100, 100, -1, -1, 100, -1, 100, 100, 100, 100, -1, -1, 100, -1, 100, 100, 100, -1, -1, -1, 100, 100, 100, 100, -1, 100, -1, 100, 100, -1, -1, 100, 100, -1, -1, 100, -1, -1, -1, -1, -1, -1],
    #         [100, -1, 100, 100, 100, -1, 100, -1, -1, 100, 100, -1, -1, -1, -1, 100, -1, -1, -1, 100, -1, -1, -1, 100, 100, -1, -1, -1, -1, 100, 100, 100, -1, -1, -1, -1, -1, 100, -1, -1, -1, -1, -1, -1, 100, -1, 100, 100, 100, -1],
    #         [100, 100, 100, -1, 100, 100, -1, 100, 100, -1, 100, -1, -1, -1, -1, 100, -1, 100, 100, 100, -1, 100, -1, -1, -1, 100, -1, -1, -1, 100, -1, 100, -1, 100, -1, 100, -1, -1, -1, 100, -1, -1, 100, -1, -1, -1, -1, 100, 100, 100],
    #         [-1, -1, -1, -1, -1, 100, -1, 100, -1, 100, 100, 100, -1, -1, -1, -1, 100, -1, 100, 100, -1, 100, 100, -1, -1, -1, -1, 100, 100, -1, 100, -1, -1, 100, -1, -1, 100, 100, 100, -1, -1, -1, -1, 100, -1, 100, 100, -1, 100, -1],
    #         [-1, -1, 100, 100, -1, -1, -1, 100, 100, -1, -1, -1, -1, -1, 100, -1, -1, -1, 100, 100, 100, -1, -1, -1, 100, 100, 100, 100, -1, -1, -1, 100, -1, 100, 100, 100, -1, 100, 100, -1, 100, -1, -1, -1, 100, 100, -1, 100, 100, -1]
    #     ]

    def create_dc_link(self):
        # matrix_size = 50
        # # 生成一个包含随机值（0和100）的矩阵
        # self.dcs_link_matrix = np.random.choice([-1, 90], size=(matrix_size, matrix_size))
        #
        # # 将矩阵转换为对称矩阵
        # self.dcs_link_matrix = np.maximum(self.dcs_link_matrix, self.dcs_link_matrix.T)
        #
        # np.savetxt('matrix.csv', self.dcs_link_matrix, delimiter=',', fmt='%d')
        self.dcs_link_matrix = np.loadtxt('matrix.csv', delimiter=',', dtype=int)



    def create_all_sfc(self):
        # sfc_installed = {}
        # item_num = 0
        # dc_num = 16
        # file_name = "../../CPU_MEMORY_LINK_RESOURCE.csv"
        # with open(file_name, 'r', newline='') as csvfile_r:
        #     reader = csv.reader(csvfile_r)
        #     for row in reader:
        #         item_num += 1
        #         cpu_request = round(float(row[9]), 6)
        #         mem_request = round(float(row[10]), 6)
        #         sfc_unit = []
        #         for i in range(int(row[7]) + 1):
        #             dc_temp = "dc" + str(random.randint(0, 15))
        #             dc_vnf_temp = "sfc" + str(item_num) + "_vnf" + str(i)
        #             sfc_unit.append([dc_temp, dc_vnf_temp, cpu_request, mem_request])
        #         sfc_installed[item_num] = sfc_unit
        #         self.add_SFC(sfc_unit)
        #         print(sfc_unit)
        #         print(item_num)
                # if item_num == 16:
                #     print(sfc_installed)
                #     break

        self.add_SFC([['dc0', 'sfc1_vnf0', 32, 12],
                        ['dc3', 'sfc1_vnf1', 22, 30],
                        ['dc6', 'sfc1_vnf2', 33, 29],
                        ['dc9', 'sfc1_vnf3', 19, 36],
                        ['dc12', 'sfc1_vnf4', 33, 35],
                        ['dc14', 'sfc1_vnf5', 21, 24]], bandwidth = 10)
        self.add_SFC([['dc0', 'sfc2_vnf0', 33, 35],
                        ['dc4', 'sfc2_vnf1', 13, 24],
                        ['dc7', 'sfc2_vnf2', 7, 31],
                        ['dc9', 'sfc2_vnf3', 25, 27],
                        ['dc13', 'sfc2_vnf4', 10, 10],
                        ['dc14', 'sfc2_vnf5', 31, 32]], bandwidth = 10)
        self.add_SFC([['dc1', 'sfc3_vnf0', 19, 33],
                        ['dc4', 'sfc3_vnf1', 2, 12],
                        ['dc8', 'sfc3_vnf2', 14, 34],
                        ['dc10', 'sfc3_vnf3', 22, 44],
                        ['dc12', 'sfc3_vnf4', 13, 25],
                        ['dc15', 'sfc3_vnf5', 26, 33]], bandwidth = 10)
        self.add_SFC([['dc2', 'sfc4_vnf0', 21, 23],
                        ['dc5', 'sfc4_vnf1', 22, 23],
                        ['dc7', 'sfc4_vnf2', 22, 32],
                        ['dc11', 'sfc4_vnf3', 22, 33],
                        ['dc13', 'sfc4_vnf4', 30, 30],
                        ['dc15', 'sfc4_vnf5', 11, 10]], bandwidth = 10)
        self.add_SFC([['dc2', 'sfc5_vnf0', 25, 25],
                        ['dc3', 'sfc5_vnf1', 22, 31],
                        ['dc6', 'sfc5_vnf2', 22, 32],
                        ['dc49', 'sfc5_vnf3', 24, 33],
                        ['dc13', 'sfc5_vnf4', 30, 31],
                        ['dc14', 'sfc5_vnf5', 33, 20]], bandwidth = 10)

        self.add_SFC([['dc16', 'evnf0', 13, 23],
                      ['dc17', 'evnf1', 22, 31],
                      ['dc18', 'evnf2', 12, 12],
                      ['dc19', 'evnf3', 24, 33],
                      ['dc20', 'evnf4', 30, 30],
                      ['dc21', 'evnf5', 34, 24],
                      ['dc22', 'evnf6', 25, 29],
                      ['dc23', 'evnf7', 39, 43],
                      ['dc24', 'evnf8', 45, 34],
                      ['dc25', 'evnf9', 34, 32],
                      ['dc26', 'evnf10', 24, 33],
                      ['dc27', 'evnf11', 49, 53],
                      ['dc28', 'evnf12', 24, 23],
                      ['dc29', 'evnf13', 43, 46],
                      ['dc30', 'evnf14', 43, 33],
                      ['dc31', 'evnf15', 43, 33],
                      ['dc32', 'evnf16', 34, 33],
                      ['dc33', 'evnf17', 56, 54],
                      ['dc34', 'evnf18', 46, 54],
                      ['dc35', 'evnf19', 54, 54],
                      ['dc36', 'evnf20', 55, 64],
                      ['dc37', 'evnf21', 45, 54],
                      ['dc38', 'evnf22', 49, 45],
                      ['dc39', 'evnf23', 34, 37],
                      ['dc40', 'evnf24', 35, 37],
                      ['dc41', 'evnf25', 35, 44],
                      ['dc42', 'evnf26', 21, 19],
                      ['dc43', 'evnf27', 43, 46],
                      ['dc44', 'evnf28', 34, 36],
                      ['dc45', 'evnf29', 44, 41],
                      ['dc46', 'evnf30', 32, 33],
                      ['dc47', 'evnf31', 44, 47],
                      ['dc48', 'evnf32', 35, 34],
                      ['dc49', 'evnf33', 34, 43]], bandwidth=10)

        # self.add_SFC([['dc0', 'dcs_vnf0', 50, 50],
        #               ['dc1', 'dcs_vnf1', 50, 50],
        #               ['dc2', 'dcs_vnf2', 50, 50],
        #               ['dc3', 'dcs_vnf3', 50, 50],
        #               ['dc4', 'dcs_vnf4', 50, 50],
        #               ['dc5', 'dcs_vnf5', 50, 50],
        #               ['dc6', 'dcs_vnf6', 50, 50],
        #               ['dc7', 'dcs_vnf7', 50, 50],
        #               ['dc8', 'dcs_vnf8', 50, 50],
        #               ['dc9', 'dcs_vnf9', 50, 50],
        #               ['dc10', 'dcs_vnf10', 50, 50],
        #               ['dc11', 'dcs_vnf11', 50, 50],
        #               ['dc12', 'dcs_vnf12', 50, 50],
        #               ['dc13', 'dcs_vnf13', 50, 50],
        #               ['dc14', 'dcs_vnf14', 50, 50],
        #               ['dc15', 'dcs_vnf15', 50, 50]])

        # if self.check_dc_threshold_exceed(0.5, 0.5):
        #     self.exceed_vnfs_dict.update(self.chooose_migrate_VNF(0.5, 0.5))
        # else:
        #     print("此时物理节点没有超过阈值！")

        # dc_temp = None
        # dc_vnf_temp = None
        # sfc_installed = {}
        # for i in range(0, sfc_number):
        #     sfc_vnf_installed = []
        #     vnf_number = random.randint(2, 5)
        #     for item in range(0, vnf_number):
        #         dc_temp = "dc" + str(random.randint(0, 15))
        #         dc_vnf_temp = "sfc" + str(i) + "_vnf" + str(item)
        #         sfc_vnf_installed.append([dc_temp, dc_vnf_temp, random.randint(10, 30), random.randint(10, 30)])
        #         # print(temp)
        #     self.add_SFC(sfc_vnf_installed)
        #     sfc_installed[i] = sfc_vnf_installed

    def create_all_sfc1(self):
        self.add_SFC([['dc13', 'dc_vnf0', 11, 17],
                      ['dc25', 'dc_vnf1', 11, 22],
                      ['dc42', 'dc_vnf4343', 32, 29]], bandwidth = 10)
        # if self.check_dc_threshold_exceed(0.5, 0.5):
        #     self.exceed_vnfs_dict.update(self.chooose_migrate_VNF(0.5, 0.5))
        # else:
        #     print("此时物理节点没有超过阈值！")

    def create_all_sfc2(self):
        self.add_SFC([['dc0', 'dc_vnf101', 25, 26],
                      ['dc3', 'dc_vnf223', 15, 20],
                    ['dc11', 'dc_vnf102', 19, 37],
                      ['dc37', 'dc_vnf103', 2, 1]], bandwidth = 10)
        # if self.check_dc_threshold_exceed(0.5, 0.5):
        #     self.exceed_vnfs_dict.update(self.chooose_migrate_VNF(0.5, 0.5))
        # else:
        #     print("此时物理节点没有超过阈值！")

    def create_all_sfc3(self):
        self.add_SFC([['dc2', 'dc_vnf222', 33, 35],
                      ['dc3', 'dc_vnf223', 15, 15],
                      ['dc18', 'dc_vnf223333', 12, 20]], bandwidth = 10)
        # if self.check_dc_threshold_exceed(0.5, 0.5):
        #     self.exceed_vnfs_dict.update(self.chooose_migrate_VNF(0.5, 0.5))
        # else:
        #     print("此时物理节点没有超过阈值！")

    def create_all_sfc4(self):
        self.add_SFC([['dc7', 'dc_vnf4', 22, 23],
                      ['dc10', 'dc_vnf5', 41, 9],
                      ['dc16', 'dc_vnf5888', 32, 33]], bandwidth = 10)
        # if self.check_dc_threshold_exceed(0.5, 0.5):
        #     self.exceed_vnfs_dict.update(self.chooose_migrate_VNF(0.5, 0.5))
        # else:
        #     print("此时物理节点没有超过阈值！")

    def create_all_sfc5(self):
        self.add_SFC([['dc1', 'dc_vnf6', 47, 33],
                      ['dc2', 'dc_vnf9', 11, 11],
                      ['dc8', 'dc_vnf7', 31, 13],
                      ['dc9', 'dc_vnf8', 14, 12],
                      ['dc16', 'dc_vnf58898', 41, 33]], bandwidth = 10)





    def create_all_sfc6(self):
        self.add_SFC([['dc1', 'dc_vnf129', 11, 14],
                      ['dc0', 'dc_vnf10', 22, 19], ], bandwidth=2)
        self.add_SFC([['dc1', 'dc_vnf11', 11, 14],
                      ['dc3', 'dc_vnf122', 21, 11], ], bandwidth=4)
        self.add_SFC([['dc8', 'dc_vnf153', 21, 23],
                      ['dc9', 'dc_vnf134', 32, 23],
                      ['dc12', 'dc_vnf184', 34, 23]], bandwidth=3)
        self.add_SFC([['dc8', 'dc_vnf1113', 21, 23],
                      ['dc9', 'dc_vnf11114', 22, 19],
                      ['dc12', 'dc_vnf12224', 34, 23]], bandwidth=2)
        self.add_SFC([['dc21', 'dc_vnf14445', 33, 43], ['dc25', 'dc_vnf16', 33, 43]], bandwidth=4)

        self.add_SFC([['dc21', 'dc_vnf17', 13, 23], ['dc25', 'dc_vnf18', 21, 12]], bandwidth=2)

        self.add_SFC([['dc26', 'dc_vnf19', 13, 23], ['dc27', 'dc_vnf20', 21, 12]], bandwidth=7)

        self.add_SFC([['dc26', 'dc_vnf21', 34, 31], ['dc27', 'dc_vnf22', 18, 19]], bandwidth=2)

        self.add_SFC([['dc28', 'dc_vnf23', 9, 11], ['dc29', 'dc_vnf24', 12, 21]], bandwidth=3)

        # self.add_SFC([['dc21', 'dc_vnf14445', random.choices(self.bono_cpu), random.choices(self.bono_mem)],
        #               ['dc25', 'dc_vnf16', random.choices(self.bono_cpu), random.choices(self.bono_mem)]], bandwidth=random.choices(self.max_bandwidth))
        #
        # self.add_SFC([['dc21', 'dc_vnf17', random.choices(self.bono_cpu), random.choices(self.bono_mem)],
        #               ['dc25', 'dc_vnf18', random.choices(self.bono_cpu), random.choices(self.bono_mem)]], bandwidth=random.choices(self.max_bandwidth))
        #
        # self.add_SFC([['dc26', 'dc_vnf19', random.choices(self.bono_cpu), random.choices(self.bono_mem)],
        #               ['dc27', 'dc_vnf20', random.choices(self.bono_cpu), random.choices(self.bono_mem)]], bandwidth=random.choices(self.max_bandwidth))
        #
        # self.add_SFC([['dc26', 'dc_vnf21', random.choices(self.bono_cpu), random.choices(self.bono_mem)],
        #               ['dc27', 'dc_vnf22', random.choices(self.bono_cpu), random.choices(self.bono_mem)]], bandwidth=random.choices(self.max_bandwidth))
        #
        # self.add_SFC([['dc28', 'dc_vnf23', random.choices(self.bono_cpu), random.choices(self.bono_mem)],
        #               ['dc29', 'dc_vnf24', random.choices(self.bono_cpu), random.choices(self.bono_mem)]], bandwidth=random.choices(self.max_bandwidth))

        self.add_SFC([['dc28', 'dc_vnf25', 33, 31],['dc29', 'dc_vnf26', 8, 12]], bandwidth=3)

        self.add_SFC([['dc30', 'dc_vnf27', 13, 15],['dc31', 'dc_vnf28', 31, 29]], bandwidth=5)

        self.add_SFC([['dc30', 'dc_vnf29', 32, 23],['dc31', 'dc_vnf30', 11, 12]], bandwidth=7)

        self.add_SFC([['dc32', 'dc_vnf31', 7, 9],['dc33', 'dc_vnf32', 12, 12]], bandwidth=5)

        self.add_SFC([['dc32', 'dc_vnf33', 32, 41],['dc33', 'dc_vnf34', 12, 7]], bandwidth=7)

        self.add_SFC([['dc34', 'dc_vnf35', 9, 12],['dc35', 'dc_vnf36', 32, 41]], bandwidth=5)

        self.add_SFC([['dc34', 'dc_vnf37', 31, 29],['dc35', 'dc_vnf38', 9, 12]], bandwidth=7)

        self.add_SFC([['dc36', 'dc_vnf39', 23, 21],['dc37', 'dc_vnf40', 23, 19]], bandwidth=5)

        self.add_SFC([['dc36', 'dc_vnf41', 11, 13],['dc37', 'dc_vnf42', 22, 24]], bandwidth=7)

        self.add_SFC([['dc38', 'dc_vnf43', 22, 21],['dc39', 'dc_vnf44', 11, 9]], bandwidth=5)

        self.add_SFC([['dc38', 'dc_vnf45', 11, 12],['dc39', 'dc_vnf46', 38, 34]], bandwidth=7)

        self.add_SFC([['dc40', 'dc_vnf47', 22, 23],['dc41', 'dc_vnf48', 11, 12]], bandwidth=5)

        self.add_SFC([['dc40', 'dc_vnf49', 22, 22],['dc41', 'dc_vnf50', 37, 41]], bandwidth=7)

        self.add_SFC([['dc42', 'dc_vnf51', 22, 25],['dc43', 'dc_vnf52', 11, 11]], bandwidth=5)

        self.add_SFC([['dc42', 'dc_vnf53', 22, 27],['dc43', 'dc_vnf54', 35, 33]], bandwidth=7)

        self.add_SFC([['dc44', 'dc_vnf55', 23, 21],['dc45', 'dc_vnf56', 12, 11]], bandwidth=5)

        self.add_SFC([['dc44', 'dc_vnf57', 9, 7],['dc45', 'dc_vnf58', 32, 41]], bandwidth=7)

        self.add_SFC([['dc46', 'dc_vnf59', 33, 32],['dc47', 'dc_vnf60', 11, 10]], bandwidth=5)

        self.add_SFC([['dc46', 'dc_vnf61', 11, 12],['dc47', 'dc_vnf62', 21, 19]], bandwidth=7)

        self.add_SFC([['dc48', 'dc_vnf63', 33, 34],['dc49', 'dc_vnf64', 12, 14]], bandwidth=5)

        self.add_SFC([['dc48', 'dc_vnf65', 11, 12],['dc49', 'dc_vnf66', 32, 27]], bandwidth=7)

        self.add_SFC([['dc15', 'dc_vnf6543', 11, 11], ['dc21', 'dc_vnf653454', 11, 11]], bandwidth=7)
    def migration_cost(self, choosed_dc, exceed_dc, need_migration_vnf):
        # sfc_temp_dict是选定vnf上下vnf所在的dc节点{3: 'dc12', 4: ('dc7', 'dc13'), 5: ('dc10', 'dc14')}
        sfc_temp_dict = {}
        need_migration_vnf_cpu_resource = 0
        need_migration_vnf_mem_resource = 0
        for sfc_number in self.SFCs:
            sfc_temp = -1
            temp = -1
            dc_vnf_list = []
            for vnf in self.SFCs[sfc_number]['sfc_vnf']:
                dc_vnf_list.append(vnf[0])
                temp = temp + 1
                if vnf[1] == need_migration_vnf and vnf[0] == exceed_dc:
                    sfc_temp = temp
                    need_migration_vnf_cpu_resource = vnf[2] + need_migration_vnf_cpu_resource
                    need_migration_vnf_mem_resource = vnf[3] + need_migration_vnf_mem_resource
            if sfc_temp!= -1:
                if sfc_temp == len(dc_vnf_list) - 1:
                    sfc_temp_dict[sfc_number] = [dc_vnf_list[sfc_temp - 1]]
                elif sfc_temp == 0:
                    sfc_temp_dict[sfc_number] = [dc_vnf_list[sfc_temp + 1]]
                else:
                    sfc_temp_dict[sfc_number] = [dc_vnf_list[sfc_temp - 1], dc_vnf_list[sfc_temp + 1]]
        link_cost = 0
        for sfc in sfc_temp_dict.keys():
            for dc in sfc_temp_dict[sfc]:
                path = self.return_dijkstra_path(self.dcs_link_matrix, int(choosed_dc.split('dc')[-1]), int(dc.split('dc')[-1]), self.SFCs[sfc]['bandwidth'])
                link_cost = len(path) * self.SFCs[sfc]['bandwidth'] + link_cost



        if need_migration_vnf_cpu_resource > self.dcs_dict[choosed_dc]['dc_cpu_resource'] or need_migration_vnf_mem_resource > self.dcs_dict[choosed_dc]['dc_mem_resource']:
            return 100
        else:
            rate = (self._dc_cpu_resourse - (self.dcs_dict[choosed_dc]['dc_cpu_resource'] - need_migration_vnf_cpu_resource)) * 0.5 + (self._dc_mem_resourse - (self.dcs_dict[choosed_dc]['dc_mem_resource'] - need_migration_vnf_mem_resource)) * 0.5
            node_cost = rate * self.dc_cost/(0.5*self._dc_cpu_resourse+0.5*self._dc_mem_resourse)
        result = link_cost/100 + node_cost/1000
        return result

    def chooose_migrate_VNF(self, exceed_dc):
        '''
        选择过载dc节点中需要迁移的节点
        这个函数是选出dc节点中资源占用最多的节点
        :param vnf_cpu_resource_percent:
        :param vnf_mem_resource_percent:
        :return:
        '''
        sfc_vnf_list = []
        for key in self.dcs_dict[exceed_dc]['vnfs'].keys():
            sfc_vnf_list.append(key)
        # sfc_vnf_list为过载节点中所有vnf的列表

        temp_dict = {}
        for vnf in sfc_vnf_list:
            for sfc_number in self.SFCs:
                for sfc_list in self.SFCs[sfc_number]['sfc_vnf']:
                    if sfc_list[1] == vnf:
                        temp_dict[sfc_number] = [vnf, self.SFCs[sfc_number]['bandwidth']]

        num_duplicates = None
        total_value = None
        temp_dc_dict = {}
        for sfc in temp_dict:
            if any(value[0] == temp_dict[sfc][0] for value in temp_dict.values()):
                num_duplicates = sum(1 for value in temp_dict.values() if value[0] == temp_dict[sfc][0])
                total_value = sum(value[1] for value in temp_dict.values() if value[0] == temp_dict[sfc][0])
            temp_dc_dict[temp_dict[sfc][0]] = [num_duplicates, total_value]
        cost_dict = {}
        for key in temp_dc_dict:
            cost_dict[key] = temp_dc_dict[key][0]/len(temp_dict) + temp_dc_dict[key][1]/self.max_bandwidth

        min_value = 10000000
        min_key = None
        # 遍历字典的值
        for key, value in cost_dict.items():
            # 如果当前值小于当前最小值，则更新最小值和对应的键
            if min_value is None or value < min_value:
                min_value = value
                min_key = key
        return min_key

    def after_migration_dc_and_link(self, choosed_dc, exceed_dc, need_migration_vnf):
        sfc_temp_dict = {}
        need_migration_vnf_cpu_resource = 0
        need_migration_vnf_mem_resource = 0
        for sfc_number in self.SFCs:
            sfc_temp = -1
            temp = -1
            dc_vnf_list = []
            for vnf in self.SFCs[sfc_number]['sfc_vnf']:
                dc_vnf_list.append(vnf[0])
                temp = temp + 1
                if vnf[1] == need_migration_vnf and vnf[0] == exceed_dc:
                    sfc_temp = temp
                    need_migration_vnf_cpu_resource = vnf[2] + need_migration_vnf_cpu_resource
                    need_migration_vnf_mem_resource = vnf[3] + need_migration_vnf_mem_resource
            if sfc_temp != -1:
                if sfc_temp == len(dc_vnf_list) - 1:
                    sfc_temp_dict[sfc_number] = [dc_vnf_list[sfc_temp - 1]]
                elif sfc_temp == 0:
                    sfc_temp_dict[sfc_number] = [dc_vnf_list[sfc_temp + 1]]
                else:
                    sfc_temp_dict[sfc_number] = [dc_vnf_list[sfc_temp - 1], dc_vnf_list[sfc_temp + 1]]
        # 首先删除原本SFC路径所需要的资源
        for sfc_name in sfc_temp_dict:
            for i in range(0, len(self.SFCs_dc_path[sfc_name]) - 1):
                self.dcs_link_matrix[self.SFCs_dc_path[sfc_name][i]][self.SFCs_dc_path[sfc_name][i + 1]] = self.dcs_link_matrix[self.SFCs_dc_path[sfc_name][i]][self.SFCs_dc_path[sfc_name][i + 1]] + self.SFCs[sfc_name]['bandwidth']
                self.dcs_link_matrix[self.SFCs_dc_path[sfc_name][i + 1]][self.SFCs_dc_path[sfc_name][i]] = self.dcs_link_matrix[self.SFCs_dc_path[sfc_name][i + 1]][self.SFCs_dc_path[sfc_name][i]] + self.SFCs[sfc_name]['bandwidth']
        # 然后替换原来self.SFCs中原来的sfc为新节点
        # print("after_migration_dc_and_link:",self.dcs_dict)
        # print("after_migration_dc_and_link:",sfc_temp_dict)
        for sfc_name1 in sfc_temp_dict:
            # 释放路径中节点资源
            for list in self.SFCs[sfc_name1]['sfc_vnf']:
                for i in self.dcs_dict:
                    if list[0] == i:
                        self.dcs_dict[i]['dc_cpu_resource'] += list[2]
                        self.dcs_dict[i]['dc_mem_resource'] += list[3]
                        del self.dcs_dict[i]['vnfs'][list[1]]

            for index, sfc_list in enumerate(self.SFCs[sfc_name1]['sfc_vnf']):
                if sfc_list[0] == exceed_dc and sfc_list[1] == need_migration_vnf:
                    self.SFCs[sfc_name1]['sfc_vnf'][index][0] = choosed_dc

            path = self.upgrate_sfc_resource(self.SFCs[sfc_name1]['sfc_vnf'], bandwidth=self.SFCs[sfc_name1]['bandwidth'])
            self.SFCs_dc_path[sfc_name1] = path

    def upgrate_sfc_resource(self, vnfs_list, **params):
        sfc_dc = []
        number_sfc_dc = []
        band_value = []
        for value in params.values():
            band_value.append(value)
        # print(band_value)
        if not self.dc_sfc_resource_check(vnfs_list=vnfs_list):
            return 0
        for item_list in vnfs_list:
            self.add_VNF(item_list[0], item_list[1], item_list[2], item_list[3])
        for sfc in vnfs_list:
            sfc_dc.append(sfc[0])
        for number in sfc_dc:
            number_sfc_dc.append(int(number.split("c")[-1]))
        # print(number_sfc_dc)
        all_dc_path = []
        for i in range(0, len(number_sfc_dc) - 1):
            path = self.dijkstra(self.dcs_link_matrix, int(number_sfc_dc[i]), int(number_sfc_dc[i + 1]),
                                 int(band_value[0]))
            all_dc_path.append(path)
        new_path_list = []
        for i in range(len(all_dc_path)):
            if i == 0:
                new_path_list.append(all_dc_path[i])
            else:
                if all_dc_path[i][0] == new_path_list[-1][-1]:
                    new_path_list[-1].extend(all_dc_path[i][1:])
                else:
                    new_path_list.append(all_dc_path[i])
        return new_path_list[0]

    def TABU_migration_cost(self, choosed_dc, exceed_dc, need_migration_vnf):
        need_migration_vnf_cpu_resource = 0
        need_migration_vnf_mem_resource = 0
        TABU_migration_cost = 0
        for sfc_number in self.SFCs:
            for vnf in self.SFCs[sfc_number]['sfc_vnf']:
                if vnf[1] == need_migration_vnf and vnf[0] == exceed_dc:
                    need_migration_vnf_cpu_resource = vnf[2] + need_migration_vnf_cpu_resource
                    need_migration_vnf_mem_resource = vnf[3] + need_migration_vnf_mem_resource

        for dc_info in self.dcs_dict:
            if dc_info == choosed_dc:
                dc_cpu_resource = self.dcs_dict[dc_info]['dc_cpu_resource'] - need_migration_vnf_cpu_resource
                dc_mem_resource = self.dcs_dict[dc_info]['dc_mem_resource'] - need_migration_vnf_mem_resource
                if dc_cpu_resource < 0 or dc_mem_resource < 0:
                    return -1
                else:
                    TABU_migration_cost = 0.5*dc_cpu_resource/self._dc_cpu_resourse + 0.5*dc_mem_resource/self._dc_mem_resourse
        return TABU_migration_cost

    def Delay_time(self, choosed_dc, exceed_dc, need_migration_vnf):
        # sfc_temp_dict是选定vnf上下vnf所在的dc节点{3: 'dc12', 4: ('dc7', 'dc13'), 5: ('dc10', 'dc14')}
        sfc_temp_dict = {}
        path = []
        need_migration_vnf_cpu_resource = 0
        need_migration_vnf_mem_resource = 0
        for sfc_number in self.SFCs:
            sfc_temp = -1
            temp = -1
            dc_vnf_list = []
            for vnf in self.SFCs[sfc_number]['sfc_vnf']:
                dc_vnf_list.append(vnf[0])
                temp = temp + 1
                if vnf[1] == need_migration_vnf and vnf[0] == exceed_dc:
                    sfc_temp = temp
                    need_migration_vnf_cpu_resource = vnf[2] + need_migration_vnf_cpu_resource
                    need_migration_vnf_mem_resource = vnf[3] + need_migration_vnf_mem_resource
            if sfc_temp!= -1:
                if sfc_temp == len(dc_vnf_list) - 1:
                    sfc_temp_dict[sfc_number] = [dc_vnf_list[sfc_temp - 1]]
                elif sfc_temp == 0:
                    sfc_temp_dict[sfc_number] = [dc_vnf_list[sfc_temp + 1]]
                else:
                    sfc_temp_dict[sfc_number] = [dc_vnf_list[sfc_temp - 1], dc_vnf_list[sfc_temp + 1]]
        for sfc in sfc_temp_dict.keys():
            for dc in sfc_temp_dict[sfc]:
                path = self.return_dijkstra_path(self.dcs_link_matrix, int(choosed_dc.split('dc')[-1]), int(dc.split('dc')[-1]), self.SFCs[sfc]['bandwidth'])
        return len(path)

    def Resource_Utilization(self, choosed_dc):
        return (0.5*self.dcs_dict[choosed_dc]['dc_cpu_resource'])/self._dc_cpu_resourse + (0.5*self.dcs_dict[choosed_dc]['dc_mem_resource'])/self._dc_mem_resourse

    def addDataCenter(self, dc_name, dc_cpu_resource, dc_mem_resource):
        self.dcs_dict[dc_name]['dc_cpu_resource'] = dc_cpu_resource
        self.dcs_dict[dc_name]['dc_mem_resource'] = dc_mem_resource
        return dc_name

    def addLink(self, graph, start, end, bandwidth):
        temp_graph = [[0] * len(graph[0]) for _ in range(len(graph))]
        for i in range(len(graph)):
            for j in range(len(graph[0])):
                if graph[i][j] > 0:
                    temp_graph[i][j] = self.max_bandwidth - graph[i][j]
                else:
                    temp_graph[i][j] = -1

        num_vertices = len(temp_graph)
        distances = [float('inf')] * num_vertices
        distances[start] = 0
        path = [None] * num_vertices
        visited = [False] * num_vertices  # 用于记录每个顶点是否已经被访问过
        priority_queue = [(0, start)]

        while priority_queue:
            current_distance, current_vertex = heapq.heappop(priority_queue)

            # 如果当前顶点已经被访问过，跳过
            if visited[current_vertex]:
                continue
            visited[current_vertex] = True

            for neighbor, weight in enumerate(temp_graph[current_vertex]):
                if weight == -1 or self.max_bandwidth - weight < bandwidth:
                    continue
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    path[neighbor] = current_vertex
                    heapq.heappush(priority_queue, (distance, neighbor))

        shortest_path = []
        current_vertex = end
        while current_vertex is not None:
            shortest_path.append(current_vertex)
            current_vertex = path[current_vertex]
        shortest_path.reverse()



if __name__ == '__main__':
    __text = text()
    __text.create_dc_link()
    __text.create_all_sfc()
    __text.create_all_sfc1()
    __text.create_all_sfc2()
    __text.create_all_sfc3()
    __text.create_all_sfc4()
    __text.create_all_sfc5()
    __text.create_all_sfc6()

    # print(__text.check_dc_threshold_exceed(0.5, 0.5))
    # print(__text.return_sfcs())
    # print(__text.return_sfc_dc_path())
    # print(len(__text.check_dc_threshold_exceed(0.5, 0.5)))
    print(__text.return_dcs_dict())
    # print(__text.after_migration_dc_and_link('dc1', 'dc14', 'sfc1_vnf5'))
    print(__text.return_sfcs())
    print(__text.TABU_migration_cost('dc14', 'dc1', 'sfc1_vnf5'))
    # print(__text.return_sfc_dc_path())
    # print(__text.return_dcs_dict())
    # print(__text.dijkstra(__text.return_dcs_link_matrix(), 1, 4, 10))
    # print(__text.return_dcs_link_matrix())
    # print(__text.chooose_migrate_VNF('dc14'))
