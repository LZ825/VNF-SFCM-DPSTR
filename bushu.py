import csv
import random
import sys
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.layers import Dropout, Dense, GRU
# import matplotlib.pyplot as plt
# import os
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import math
import copy



class text():

    def __init__(self):
        self._dc_cpu_resourse = 120
        self._dc_mem_resourse = 120
        self.exceed_vnfs_dict = {}
        self.dcs_exceed_list = []
        self.dcs_dict = self.create_dc_topology(dc_number = 16)
        self.dcs_link_matrix = []
        self.installed_id = 0
        self.SFCs = {}

    def return_sfcs(self):
        return self.SFCs

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
        """
        两个节点之间最短路径算法，并计算了两个节点之间的带宽
        :param start:
        :param end:
        :param bandwidth:
        :return: 返回两个节点之间经过的节点
        """
        distances = {node: sys.maxsize for node in range(len(graph))}
        distances[start] = 0
        visited = set()
        path = {}  # 用于保存节点之间的路径

        while len(visited) < len(graph):
            min_distance = sys.maxsize
            min_node = None
            for node in range(len(graph)):
                if node not in visited and distances[node] < min_distance:
                    min_distance = distances[node]
                    min_node = node

            if min_node is None:
                break

            visited.add(min_node)

            for neighbor in range(len(graph)):
                if neighbor not in visited and graph[min_node][neighbor] != -1:
                    new_distance = distances[min_node] + graph[min_node][neighbor]
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        # 更新路径字典
                        path[neighbor] = min_node

        # 从终点反向构建最短路径
        shortest_path = [end]
        current_node = end
        while current_node != start:
            if current_node not in path:
                return None  # 无法到达目标节点
            current_node = path[current_node]
            shortest_path.append(current_node)
        shortest_path.reverse()
        for i in range(0, len(shortest_path) - 1):
            self.dcs_link_matrix[shortest_path[i]][shortest_path[i + 1]] = self.dcs_link_matrix[shortest_path[i]][shortest_path[i + 1]] - bandwidth
            self.dcs_link_matrix[shortest_path[i + 1]][shortest_path[i]] = self.dcs_link_matrix[shortest_path[i + 1]][shortest_path[i]] - bandwidth
        return shortest_path

    def return_dijkstra_path(self, graph, start, end):
        distances = {node: sys.maxsize for node in range(len(graph))}
        distances[start] = 0
        visited = set()
        path = {}  # 用于保存节点之间的路径

        while len(visited) < len(graph):
            min_distance = sys.maxsize
            min_node = None
            for node in range(len(graph)):
                if node not in visited and distances[node] < min_distance:
                    min_distance = distances[node]
                    min_node = node

            if min_node is None:
                break

            visited.add(min_node)

            for neighbor in range(len(graph)):
                if neighbor not in visited and graph[min_node][neighbor] != -1:
                    new_distance = distances[min_node] + graph[min_node][neighbor]
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        # 更新路径字典
                        path[neighbor] = min_node

        # 从终点反向构建最短路径
        shortest_path = [end]
        current_node = end
        while current_node != start:
            if current_node not in path:
                return None  # 无法到达目标节点
            current_node = path[current_node]
            shortest_path.append(current_node)
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
        for i in range(0, len(number_sfc_dc)-1):
            self.dijkstra(self.dcs_link_matrix, int(number_sfc_dc[i]), int(number_sfc_dc[i + 1]), int(band_value[0]))
            # print(self.dcs_link_matrix)
        tem_dict = {"sfc_vnf":vnfs_list}
        tem_dict.update(params)
        self.installed_id += 1
        self.SFCs[self.installed_id] = tem_dict
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

    def fitness_migration(self, position, dc_name):
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
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, int(sfc[i].split('dc')[-1]), int(sfc[i+1].split('dc')[-1]))
                        for j in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[j]][shortest_path[j + 1]] = self.dcs_link_matrix[shortest_path[j]][shortest_path[j + 1]] + vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[j + 1]][shortest_path[j]] = self.dcs_link_matrix[shortest_path[j + 1]][shortest_path[j]] + vnf_gnt_dict[result_min]
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, position, int(sfc[i+1].split('dc')[-1]))
                        # print(shortest_path)
                        for k in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[k]][shortest_path[k + 1]] = self.dcs_link_matrix[shortest_path[k]][shortest_path[k + 1]] - vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[k + 1]][shortest_path[k]] = self.dcs_link_matrix[shortest_path[k + 1]][shortest_path[k]] - vnf_gnt_dict[result_min]
                    if(temp==len(sfc)-1):
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, int(sfc[i-1].split('dc')[-1]), int(sfc[i].split('dc')[-1]))
                        # print(shortest_path)
                        for j in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[j]][shortest_path[j + 1]] = self.dcs_link_matrix[shortest_path[j]][shortest_path[j + 1]] + vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[j + 1]][shortest_path[j]] = self.dcs_link_matrix[shortest_path[j + 1]][shortest_path[j]] + vnf_gnt_dict[result_min]
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix,  position, int(sfc[i - 1].split('dc')[-1]))
                        # print(shortest_path)
                        for k in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[k]][shortest_path[k + 1]] = self.dcs_link_matrix[shortest_path[k]][shortest_path[k + 1]] - vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[k + 1]][shortest_path[k]] = self.dcs_link_matrix[shortest_path[k + 1]][shortest_path[k]] - vnf_gnt_dict[result_min]
                    if(temp!=len(sfc)-1 and temp!=0):
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, int(sfc[i].split('dc')[-1]),int(sfc[i + 1].split('dc')[-1]))
                        for j in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[j]][shortest_path[j + 1]] = self.dcs_link_matrix[shortest_path[j]][shortest_path[j + 1]] + vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[j + 1]][shortest_path[j]] = self.dcs_link_matrix[shortest_path[j + 1]][shortest_path[j]] + vnf_gnt_dict[result_min]
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, position, int(sfc[i + 1].split('dc')[-1]))
                        for k in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[k]][shortest_path[k + 1]] = self.dcs_link_matrix[shortest_path[k]][shortest_path[k + 1]] - vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[k + 1]][shortest_path[k]] = self.dcs_link_matrix[shortest_path[k + 1]][shortest_path[k]] - vnf_gnt_dict[result_min]

                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, int(sfc[i - 1].split('dc')[-1]),int(sfc[i].split('dc')[-1]))
                        for l in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[l]][shortest_path[l + 1]] = self.dcs_link_matrix[shortest_path[l]][shortest_path[l + 1]] + vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[l + 1]][shortest_path[l]] = self.dcs_link_matrix[shortest_path[l + 1]][shortest_path[l]] + vnf_gnt_dict[result_min]
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, position, int(sfc[i - 1].split('dc')[-1]))
                        for m in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[m]][shortest_path[m + 1]] = self.dcs_link_matrix[shortest_path[m]][shortest_path[m + 1]] - vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[m + 1]][shortest_path[m]] = self.dcs_link_matrix[shortest_path[m + 1]][shortest_path[m]] - vnf_gnt_dict[result_min]

        variance_cpu_resource, variance_mem_resource = self.return_dcs_resource_variance()
        fitness = 0.5 * variance_cpu_resource + 0.5 * variance_mem_resource

        self.exceed_vnfs_dict = temp_exceed_vnfs_dict
        self.dcs_exceed_list = temp_dcs_exceed_list
        self.dcs_dict = temp_dcs_dict
        self.dcs_link_matrix = temp_dcs_link_matrix
        self.installed_id = temp_installed_id
        self.SFCs = temp_SFCs


        return fitness

    def migration(self, position, dc_name):
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
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, int(sfc[i].split('dc')[-1]),
                                                                  int(sfc[i + 1].split('dc')[-1]))
                        for j in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[j]][shortest_path[j + 1]] = \
                            self.dcs_link_matrix[shortest_path[j]][shortest_path[j + 1]] + vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[j + 1]][shortest_path[j]] = \
                            self.dcs_link_matrix[shortest_path[j + 1]][shortest_path[j]] + vnf_gnt_dict[result_min]
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, position,
                                                                  int(sfc[i + 1].split('dc')[-1]))
                        # print(shortest_path)
                        for k in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[k]][shortest_path[k + 1]] = \
                            self.dcs_link_matrix[shortest_path[k]][shortest_path[k + 1]] - vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[k + 1]][shortest_path[k]] = \
                            self.dcs_link_matrix[shortest_path[k + 1]][shortest_path[k]] - vnf_gnt_dict[result_min]
                    if (temp == len(sfc) - 1):
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, int(sfc[i - 1].split('dc')[-1]),
                                                                  int(sfc[i].split('dc')[-1]))
                        # print(shortest_path)
                        for j in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[j]][shortest_path[j + 1]] = \
                            self.dcs_link_matrix[shortest_path[j]][shortest_path[j + 1]] + vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[j + 1]][shortest_path[j]] = \
                            self.dcs_link_matrix[shortest_path[j + 1]][shortest_path[j]] + vnf_gnt_dict[result_min]
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, position,
                                                                  int(sfc[i - 1].split('dc')[-1]))
                        # print(shortest_path)
                        for k in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[k]][shortest_path[k + 1]] = \
                            self.dcs_link_matrix[shortest_path[k]][shortest_path[k + 1]] - vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[k + 1]][shortest_path[k]] = \
                            self.dcs_link_matrix[shortest_path[k + 1]][shortest_path[k]] - vnf_gnt_dict[result_min]
                    if (temp != len(sfc) - 1 and temp != 0):
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, int(sfc[i].split('dc')[-1]),
                                                                  int(sfc[i + 1].split('dc')[-1]))
                        for j in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[j]][shortest_path[j + 1]] = \
                            self.dcs_link_matrix[shortest_path[j]][shortest_path[j + 1]] + vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[j + 1]][shortest_path[j]] = \
                            self.dcs_link_matrix[shortest_path[j + 1]][shortest_path[j]] + vnf_gnt_dict[result_min]
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, position,
                                                                  int(sfc[i + 1].split('dc')[-1]))
                        for k in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[k]][shortest_path[k + 1]] = \
                            self.dcs_link_matrix[shortest_path[k]][shortest_path[k + 1]] - vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[k + 1]][shortest_path[k]] = \
                            self.dcs_link_matrix[shortest_path[k + 1]][shortest_path[k]] - vnf_gnt_dict[result_min]

                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, int(sfc[i - 1].split('dc')[-1]),
                                                                  int(sfc[i].split('dc')[-1]))
                        for l in range(0, len(shortest_path) - 1):
                            self.dcs_link_matrix[shortest_path[l]][shortest_path[l + 1]] = \
                            self.dcs_link_matrix[shortest_path[l]][shortest_path[l + 1]] + vnf_gnt_dict[result_min]
                            self.dcs_link_matrix[shortest_path[l + 1]][shortest_path[l]] = \
                            self.dcs_link_matrix[shortest_path[l + 1]][shortest_path[l]] + vnf_gnt_dict[result_min]
                        shortest_path = self.return_dijkstra_path(self.dcs_link_matrix, position,
                                                                  int(sfc[i - 1].split('dc')[-1]))
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
                print("物理节点"+key+"过载！！！！！！")

        self.dcs_exceed_list = dc_threshold_exceed_list
        return dc_threshold_exceed_list


    def return_migration_cost(self):
        return 0

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



    def clear_exceed_vnfs_dict(self):
        self.exceed_vnfs_dict.clear()

    def creaate_dc_link(self):
        self.dcs_link_matrix = [
            [-1, 100, -1, -1, -1, -1, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [100, -1, -1, -1, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, 100, -1, -1, 100, 100, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, 100, 100, -1, -1, -1, -1, 100, -1, -1, -1, -1],
            [-1, 100, 100, -1, -1, -1, 100, -1, -1, 100, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, 100, -1, -1, -1, -1, -1, -1, -1, 100, -1, -1, -1, -1],
            [100, -1, -1, 100, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 100, -1, -1],
            [-1, -1, 100, -1, -1, -1, -1, -1, -1, 100, -1, 100, -1, 100, -1, -1],
            [-1, -1, -1, -1, 100, -1, -1, -1, 100, -1, 100, -1, -1, -1, -1, 100],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 100, -1, -1, 100, -1, 100, -1],
            [-1, -1, -1, 100, -1, 100, -1, -1, 100, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 100, -1, -1, -1, 100, -1],
            [-1, -1, -1, -1, -1, -1, -1, 100, 100, -1, -1, -1, -1, -1, -1, 100],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 100, -1, 100, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 100, -1, -1, -1, 100, -1, -1]
        ]

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

    def chooose_migrate_VNF(self, vnf_cpu_resource_percent, vnf_mem_resource_percent):
        '''
        选择过载dc节点中需要迁移的节点
        这个函数是选出dc节点中资源占用最多的节点
        :param vnf_cpu_resource_percent:
        :param vnf_mem_resource_percent:
        :return:
        '''
        need_dc_dict = {}
        vnf_list = []
        migrate_vnf_dc_list = self.check_dc_threshold_exceed(0.5, 0.5)

        # for dc in migrate_vnf_dc_list:
        #     max_vnf_i = list(self.dcs_dict[dc]['vnfs'])[0]
        #     for vnf_i in self.dcs_dict[dc]['vnfs']:
        #         if vnf_cpu_resource_percent * self.dcs_dict[dc]['vnfs'][vnf_i]['vnf_cpu_resource'] +\
        #                 vnf_mem_resource_percent * self.dcs_dict[dc]['vnfs'][vnf_i]['vnf_mem_resource'] >\
        #                 vnf_cpu_resource_percent * self.dcs_dict[dc]['vnfs'][max_vnf_i]['vnf_cpu_resource'] +\
        #                 vnf_mem_resource_percent * self.dcs_dict[dc]['vnfs'][max_vnf_i]['vnf_mem_resource']:
        #             max_vnf_i = vnf_i
        #     vnf_list.append(max_vnf_i)
        # print("migrate_vnf_dc_list:{}".format(migrate_vnf_dc_list))
        for value in migrate_vnf_dc_list:
            vnfs_dict = self.dcs_dict[value]['vnfs']
            # print(vnfs_dict)
            vnfs_dict_list = []
            for key in vnfs_dict.keys():
                vnfs_dict_list.append(key)
            maxkey = vnfs_dict_list[0]
            for key in vnfs_dict.keys():
                # print(maxkey)
                if vnf_cpu_resource_percent*vnfs_dict[key]['vnf_cpu_resource'] + vnf_mem_resource_percent*vnfs_dict[key]['vnf_mem_resource'] > vnf_cpu_resource_percent*vnfs_dict[maxkey]['vnf_cpu_resource'] + vnf_mem_resource_percent*vnfs_dict[maxkey]['vnf_mem_resource']:
                    maxkey = key
            vnf_list.append({'vnf_name': maxkey, **vnfs_dict[maxkey]})
        # print("vnf_list:{}".format(vnf_list))
        for i in range(0, len(vnf_list)):
            need_dc_dict[migrate_vnf_dc_list[i]] = vnf_list[i]
        # print("need_dc_dict:{}".format(need_dc_dict))
        return need_dc_dict