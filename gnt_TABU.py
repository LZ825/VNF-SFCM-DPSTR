import numpy as np
import gnt_Util
import time
import random
# 定义禁忌搜索的VNF迁移算法tabu_search_vnf_migration
def DPSTR(text, vnf_mapping, exceed_dc, tabu_list_size, max_iterations):
    current_solution = vnf_mapping
    best_solution = vnf_mapping
    tabu_list = []
    need_migration_vnf = text1.chooose_migrate_VNF(exceed_dc)

    current_cost = calculate_cost(text, current_solution, exceed_dc, need_migration_vnf)
    best_cost = current_cost

    for iteration in range(max_iterations):
        neighbors = generate_neighbors(current_solution, exceed_dc)
        best_neighbor = 0
        best_neighbor_cost = 0
        for neighbor in neighbors:
            if neighbor not in tabu_list:
                neighbor_cost = calculate_cost(text, neighbor, exceed_dc, need_migration_vnf)
                if neighbor_cost > best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor_cost

        current_solution = best_neighbor
        current_cost = best_neighbor_cost
        tabu_list.append(best_neighbor)
        if len(tabu_list) > tabu_list_size:
            tabu_list.pop(0)
        if current_cost > best_cost:
            best_solution = current_solution
            best_cost = current_cost

    return best_solution, need_migration_vnf


# 计算目标函数值
# def calculate_cost(text, vnf_mapping, exceed_dc, need_migration_vnf):
#     vnf_mapping = "dc" + str(vnf_mapping)
#     if 'dc' + str(vnf_mapping) != exceed_dc:
#         fitness = text.TABU_migration_cost(vnf_mapping, exceed_dc, need_migration_vnf)
#         # print("59  calculate_cost:", fitness)
#         return fitness
#     else:
#         return -1

def calculate_cost(text, choosed_dc, exceed_dc, need_migration_vnf):
    choosed_dc = "dc" + str(choosed_dc)
    if choosed_dc!= exceed_dc:
        return text.migration_cost(choosed_dc, exceed_dc, need_migration_vnf)
    else:
        return -1

# 生成当前解的邻域解
def generate_neighbors(vnf_mapping, exceed_dc):
    neighbors = []
    for i in range(0, 6):
        particles_position_binary_list = decimal_to_binary(vnf_mapping, 6)
        for j in range(0, 6):
            if np.random.rand() < 0.5:  # 生成一个0到1之间的随机数，如果小于mutation_prob，则进行翻转
                particles_position_binary_list[j] = 1 - particles_position_binary_list[j]  # 进行二进制位的翻转
        dc_local = binary_to_decimal(particles_position_binary_list)  # 将二进制编码转换回十进制值
        while (dc_local > 49 or dc_local == int(exceed_dc.split("dc")[-1])):
            for j in range(0, 6):
                if np.random.rand() < 0.5:  # 生成一个0到1之间的随机数，如果小于mutation_prob，则进行翻转
                    particles_position_binary_list[j] = 1 - particles_position_binary_list[j]  # 进行二进制位的翻转
            dc_local = binary_to_decimal(particles_position_binary_list)  # 将二进制编码转换回十进制值
        neighbors.append(dc_local)
    return neighbors

def decimal_to_binary(decimal_value, num_bits):
    """
    将十进制值转换为二进制编码

    参数：
    decimal_value (float) - 十进制值
    num_bits (int) - 二进制编码的位数

    返回：
    binary_code (list) - 二进制编码列表
    """
    binary_code = []
    binary_str = np.binary_repr(decimal_value, num_bits)  # 使用numpy的binary_repr函数将十进制值转换为二进制字符串
    for bit in binary_str:
        binary_code.append(int(bit))  # 将二进制字符串中的每个字符转换为整数，得到二进制编码列表
    return binary_code

def binary_to_decimal(binary_code):
    """
    将二进制编码转换为十进制值

    参数：
    binary_code (list) - 二进制编码列表

    返回：
    decimal_value (float) - 十进制值
    """
    binary_str = ''.join(str(bit) for bit in binary_code)  # 将二进制编码列表转换为字符串
    decimal_value = int(binary_str, 2)  # 使用int函数将二进制字符串转换为十进制值
    return decimal_value

# def distance_last_vnf_in_dc(test_bandaim, sfcs, topology, vnf_name, dc_name):
#     '''当前vnf所在物理节点与服务功能链上上一个vnf所在物理节点的hop距离'''
#     for vnf_sequence in sfcs.keys():
#         dc_vnf_number = -1
#         for dc_vnf in sfcs[vnf_sequence]["sfc_vnf"]:
#             dc_vnf_number = dc_vnf_number + 1
#             if dc_vnf[1] == vnf_name and sfcs[vnf_sequence]["sfc_vnf"][0] == dc_vnf:
#                 distance_last_vnf = 0
#                 # print("这是服务功能链中第一个vnf")
#                 return distance_last_vnf
#             if dc_vnf[1] == vnf_name and sfcs[vnf_sequence]["sfc_vnf"][0] != dc_vnf:
#                 dc1 = sfcs[vnf_sequence]["sfc_vnf"][dc_vnf_number - 1][0]
#                 dc2 = dc_name
#                 path = test_bandaim.return_dijkstra_path(topology, int(dc1.split("dc")[1]), int(dc2.split("dc")[1]))
#                 distance_last_vnf = len(path)
#                 return distance_last_vnf
#             else:
#                 continue
#
# def distance_next_vnf_in_dc(test_bandaim, sfcs, topology, vnf_name, dc_name):
#     '''当前vnf所在物理节点与服务功能链上下一个vnf所在物理节点的hop距离'''
#     for vnf_sequence in sfcs.keys():
#         dc_vnf_number = -1
#         for dc_vnf in sfcs[vnf_sequence]["sfc_vnf"]:
#             dc_vnf_number = dc_vnf_number + 1
#             if dc_vnf[1] == vnf_name and sfcs[vnf_sequence]["sfc_vnf"][-1] == dc_vnf:
#                 distance_last_vnf = 0
#                 # print("这是服务功能链中最后一个vnf")
#                 return distance_last_vnf
#             if dc_vnf[1] == vnf_name and sfcs[vnf_sequence]["sfc_vnf"][-1] != dc_vnf:
#                 dc1 = sfcs[vnf_sequence]["sfc_vnf"][dc_vnf_number + 1][0]
#                 dc2 = dc_name
#                 path = test_bandaim.return_dijkstra_path(topology, int(dc1.split("dc")[1]), int(dc2.split("dc")[1]))
#                 distance_last_vnf = len(path)
#                 return distance_last_vnf
#             else:
#                 continue

choose_migrate_vnf = {}
if __name__ == '__main__':
    start_time = time.time()
    text1 = gnt_Util.text()
    text1.create_dc_link()
    text1.create_all_sfc()
    text1.create_all_sfc1()
    text1.create_all_sfc2()
    text1.create_all_sfc3()
    text1.create_all_sfc4()
    text1.create_all_sfc5()
    text1.create_all_sfc6()
    dicts = text1.return_dcs_dict()
    migration_cost = []

    # 设置禁忌表大小和最大迭代次数
    tabu_list_size = 5
    max_iterations = 10
    initial_mapping = random.randint(0, 49)
    fitness_list = []
    resource_variance_list = []
    migration_number = 0
    exceed_dc_list = text1.check_dc_threshold_exceed(0.5, 0.5)
    print(len(exceed_dc_list))
    print("需要迁移的dc_list：", len(exceed_dc_list))
    while len(exceed_dc_list) > 0:
        print("11111111111111111111111111111111",text1.return_dcs_dict())
        print("需要迁移的dc：", exceed_dc_list[0])
        choosed_dc, need_migration_vnf = tabu_search_vnf_migration(text1, initial_mapping, exceed_dc_list[0], tabu_list_size, max_iterations)
        choosed_dc = "dc" + str(choosed_dc)
        print("choosed_dc:",choosed_dc)
        text1.after_migration_dc_and_link(choosed_dc, exceed_dc_list[0], need_migration_vnf)
        exceed_dc_list.pop(0)
        exceed_dc_list = text1.check_dc_threshold_exceed(0.5, 0.5)
        if len(exceed_dc_list) > 0:
            if choosed_dc == exceed_dc_list[0]:
                exceed_dc_list.pop(0)
        print("列表长度为：", len(exceed_dc_list))
        print("22222222222222222222222222222222", text1.return_dcs_dict())
        migration_number += 1

    #
    # for key in exceed_dcs:
    #     vnf_dict = text1.GNT_VNF_choose(key)
    #     result_min = min(vnf_dict, key=lambda x: vnf_dict[x])
    #     best_mapping, best_cost = tabu_search_vnf_migration(text1, initial_mapping, key, tabu_list_size, max_iterations)


    end_time = time.time()
    print("程序运行时间:", end_time - start_time)

    print(resource_variance_list)
    print(migration_cost)
