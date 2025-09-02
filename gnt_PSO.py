import numpy as np
import gnt_Util
import time

def dbpso(text1, exceed_dc, num_particles, max_iterations, w, c1, c2):
    """
    离散二进制粒子群优化算法（DBPSO）的实现

    参数:
        binary_func (function): 二进制目标函数，输入为二进制位置，返回适应度值
        num_dims (int): 搜索空间的维度
        max_iterations (int): 最大迭代次数
        w (float): 惯性权重
        c1 (float): 个体学习因子
        c2 (float): 群体学习因子

    返回:
        best_position (ndarray): 最优解的二进制位置
        best_fitness (float): 最优解对应的适应度值
    """

    # 初始化粒子群的位置和速度
    particles_position = [0, 1, 2, 3]
    particles_velocity = [0, 0, 0, 0]

    # 初始化个体最优位置和群体最优位置
    particles_best_position = [0, 0, 0, 0]
    particles_best_fitness = [0, 0, 0, 0]
    global_best_position = 0
    global_best_fitness = 0
    new_positions = [0, 0, 0, 0]

    # 迭代更新
    for iteration in range(max_iterations):
        for i in range(0, num_particles):
            # 计算当前位置的适应度值
            fitness = binary_func(text1, particles_position[i], exceed_dc)
            # print(particles_position[i])
            # print(fitness)

            # 更新个体最优位置和群体最优位置
            if fitness > particles_best_fitness[i]:
                particles_best_position[i] = particles_position[i]
                particles_best_fitness[i] = fitness

            if fitness > global_best_fitness:
                global_best_position = particles_position[i]
                global_best_fitness = fitness

            # 更新速度和位置

            particles_velocity[i] = sigmod_int(1 / (1 + np.exp(-(
                        w * particles_velocity[i] + c1 * (particles_best_position[i] - particles_position[i]) + c2 * (
                            global_best_position - particles_position[i])))))
            particles_position[i] = 15 - particles_velocity[i]
            # print(particles_position[i])
        # print("第{}轮最优解的位置:{}".format(iteration, global_best_position))
        # print("第{}轮最优解的适应度值:{}".format(iteration, global_best_fitness))
    best_position = global_best_position
    best_fitness = global_best_fitness
    return best_position, best_fitness

# 示例：定义一个简单的二进制目标函数，输入为二进制位置，返回适应度值
def binary_func(text1, position, exceed_dc):
    if 'dc'+str(position) != exceed_dc:
        fitness = text1.fitness_migration(position, exceed_dc)
        return fitness
    else:
        return 0

def sigmod_int(sigmod_value):
    if sigmod_value <= 0.0625:
        return 0
    elif sigmod_value > 0.0625 and sigmod_value <= 0.125:
        return 1
    elif sigmod_value > 0.125 and sigmod_value <= 0.1875:
        return 2
    elif sigmod_value > 0.1875 and sigmod_value <= 0.25:
        return 3
    elif sigmod_value > 0.25 and sigmod_value <= 0.3125:
        return 4
    elif sigmod_value > 0.3125 and sigmod_value <= 0.375:
        return 5
    elif sigmod_value > 0.375 and sigmod_value <= 0.4375:
        return 6
    elif sigmod_value > 0.4375 and sigmod_value <= 0.5:
        return 7
    elif sigmod_value > 0.5 and sigmod_value <= 0.5625:
        return 8
    elif sigmod_value > 0.5625 and sigmod_value <= 0.625:
        return 9
    elif sigmod_value > 0.625 and sigmod_value <= 0.6875:
        return 10
    elif sigmod_value > 0.6875 and sigmod_value <= 0.75:
        return 11
    elif sigmod_value > 0.75 and sigmod_value <= 0.8125:
        return 12
    elif sigmod_value > 0.8125 and sigmod_value <= 0.875:
        return 13
    elif sigmod_value > 0.875 and sigmod_value <= 0.9375:
        return 14
    elif sigmod_value > 0.9375 and sigmod_value <= 1:
        return 15
    else:
        print('sigmod_int_range wrong')
        return -1

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

def distance_last_vnf_in_dc(test_bandaim, sfcs, topology, vnf_name, dc_name):
    '''当前vnf所在物理节点与服务功能链上上一个vnf所在物理节点的hop距离'''
    for vnf_sequence in sfcs.keys():
        dc_vnf_number = -1
        for dc_vnf in sfcs[vnf_sequence]["sfc_vnf"]:
            dc_vnf_number = dc_vnf_number + 1
            if dc_vnf[1] == vnf_name and sfcs[vnf_sequence]["sfc_vnf"][0] == dc_vnf:
                distance_last_vnf = 0
                # print("这是服务功能链中第一个vnf")
                return distance_last_vnf
            if dc_vnf[1] == vnf_name and sfcs[vnf_sequence]["sfc_vnf"][0] != dc_vnf:
                dc1 = sfcs[vnf_sequence]["sfc_vnf"][dc_vnf_number - 1][0]
                dc2 = dc_name
                path = test_bandaim.return_dijkstra_path(topology, int(dc1.split("dc")[1]), int(dc2.split("dc")[1]))
                distance_last_vnf = len(path)
                return distance_last_vnf
            else:
                continue

def distance_next_vnf_in_dc(test_bandaim, sfcs, topology, vnf_name, dc_name):
    '''当前vnf所在物理节点与服务功能链上下一个vnf所在物理节点的hop距离'''
    for vnf_sequence in sfcs.keys():
        dc_vnf_number = -1
        for dc_vnf in sfcs[vnf_sequence]["sfc_vnf"]:
            dc_vnf_number = dc_vnf_number + 1
            if dc_vnf[1] == vnf_name and sfcs[vnf_sequence]["sfc_vnf"][-1] == dc_vnf:
                distance_last_vnf = 0
                # print("这是服务功能链中最后一个vnf")
                return distance_last_vnf
            if dc_vnf[1] == vnf_name and sfcs[vnf_sequence]["sfc_vnf"][-1] != dc_vnf:
                dc1 = sfcs[vnf_sequence]["sfc_vnf"][dc_vnf_number + 1][0]
                dc2 = dc_name
                path = test_bandaim.return_dijkstra_path(topology, int(dc1.split("dc")[1]), int(dc2.split("dc")[1]))
                distance_last_vnf = len(path)
                return distance_last_vnf
            else:
                continue

# 调用 DBPSO 算法
choose_dbpso_vnf = {}
if __name__ == '__main__':
    start_time = time.time()
    text1 = gnt_Util.text()
    text1.create_dc_link()
    text1.create_all_sfc()
    dicts = text1.return_dcs_dict()
    migration_cost = []


    num_particles = 4
    num_dims = 50
    max_iterations = 10
    w = 0.1
    c1 = 0.4
    c2 = 1
    exceed_dcs = text1.check_dc_threshold_exceed(0.5, 0.5)
    print(exceed_dcs)
    fitness_list = []
    resource_variance_list = []
    for key in exceed_dcs:
        vnf_dict = text1.GNT_VNF_choose(key)
        result_min = min(vnf_dict, key=lambda x: vnf_dict[x])
        best_position, best_fitness = dbpso(text1, key, num_particles, max_iterations, w, c1, c2)
        print("--------------------------------------------------------------------------------------------")
        print("最优解的位置：dc", best_position)
        print("最优解的适应度值：", 1/best_fitness)
        print("--------------------------------------------------------------------------------------------")
        text1.migration(best_position, key)
        distance = distance_next_vnf_in_dc(text1, text1.return_sfcs(), text1.return_dcs_link_matrix(), result_min,
                                           'dc' + str(best_position)) + distance_last_vnf_in_dc(text1,
                                                                                               text1.return_sfcs(),
                                                                                               text1.return_dcs_link_matrix(),
                                                                                               result_min,
                                                                                               'dc' + str(best_position))
        resource_cpu_mem = (dicts['dc' + str(best_position)]['dc_cpu_resource'] + dicts['dc' + str(best_position)][
            'dc_mem_resource']) / 100
        migration_cost.append(distance * resource_cpu_mem)
        variance_cpu_resource, variance_mem_resource = text1.return_dcs_resource_variance()
        resource_variance = 0.5 * variance_cpu_resource + 0.5 * variance_mem_resource
        resource_variance_list.append(1 / resource_variance)
        fitness_list.append(1/best_fitness)


    text1.create_all_sfc1()
    exceed_dcs = text1.check_dc_threshold_exceed(0.5, 0.5)
    print(exceed_dcs)
    for key in exceed_dcs:
        vnf_dict = text1.GNT_VNF_choose(key)
        result_min = min(vnf_dict, key=lambda x: vnf_dict[x])
        best_position, best_fitness = dbpso(text1, key, num_particles, max_iterations, w, c1, c2)
        print("--------------------------------------------------------------------------------------------")
        print("最优解的位置：dc", best_position)
        print("最优解的适应度值：", 1/best_fitness)
        print("--------------------------------------------------------------------------------------------")
        text1.migration(best_position, key)
        distance = distance_next_vnf_in_dc(text1, text1.return_sfcs(), text1.return_dcs_link_matrix(), result_min,
                                           'dc' + str(best_position)) + distance_last_vnf_in_dc(text1,
                                                                                                text1.return_sfcs(),
                                                                                                text1.return_dcs_link_matrix(),
                                                                                                result_min,
                                                                                                'dc' + str(
                                                                                                    best_position))
        resource_cpu_mem = (dicts['dc' + str(best_position)]['dc_cpu_resource'] + dicts['dc' + str(best_position)][
            'dc_mem_resource']) / 100
        migration_cost.append(distance * resource_cpu_mem)
        variance_cpu_resource, variance_mem_resource = text1.return_dcs_resource_variance()
        resource_variance = 0.5 * variance_cpu_resource + 0.5 * variance_mem_resource
        resource_variance_list.append(1 / resource_variance)
        fitness_list.append(1/best_fitness)

    text1.create_all_sfc2()
    exceed_dcs = text1.check_dc_threshold_exceed(0.5, 0.5)
    print(exceed_dcs)
    for key in exceed_dcs:
        vnf_dict = text1.GNT_VNF_choose(key)
        result_min = min(vnf_dict, key=lambda x: vnf_dict[x])
        best_position, best_fitness = dbpso(text1, key, num_particles, max_iterations, w, c1, c2)
        print("--------------------------------------------------------------------------------------------")
        print("最优解的位置：dc", best_position)
        print("最优解的适应度值：", 1/best_fitness)
        print("--------------------------------------------------------------------------------------------")
        text1.migration(best_position, key)
        distance = distance_next_vnf_in_dc(text1, text1.return_sfcs(), text1.return_dcs_link_matrix(), result_min,
                                           'dc' + str(best_position)) + distance_last_vnf_in_dc(text1,
                                                                                                text1.return_sfcs(),
                                                                                                text1.return_dcs_link_matrix(),
                                                                                                result_min,
                                                                                                'dc' + str(
                                                                                                    best_position))
        resource_cpu_mem = (dicts['dc' + str(best_position)]['dc_cpu_resource'] + dicts['dc' + str(best_position)][
            'dc_mem_resource']) / 100
        migration_cost.append(distance * resource_cpu_mem)
        variance_cpu_resource, variance_mem_resource = text1.return_dcs_resource_variance()
        resource_variance = 0.5 * variance_cpu_resource + 0.5 * variance_mem_resource
        resource_variance_list.append(1 / resource_variance)
        fitness_list.append(1/best_fitness)

    text1.create_all_sfc3()
    exceed_dcs = text1.check_dc_threshold_exceed(0.5, 0.5)
    print(exceed_dcs)
    for key in exceed_dcs:
        vnf_dict = text1.GNT_VNF_choose(key)
        result_min = min(vnf_dict, key=lambda x: vnf_dict[x])
        best_position, best_fitness = dbpso(text1, key, num_particles, max_iterations, w, c1, c2)
        print("--------------------------------------------------------------------------------------------")
        print("最优解的位置：dc", best_position)
        print("最优解的适应度值：", 1/best_fitness)
        print("--------------------------------------------------------------------------------------------")
        text1.migration(best_position, key)
        distance = distance_next_vnf_in_dc(text1, text1.return_sfcs(), text1.return_dcs_link_matrix(), result_min,
                                           'dc' + str(best_position)) + distance_last_vnf_in_dc(text1,
                                                                                                text1.return_sfcs(),
                                                                                                text1.return_dcs_link_matrix(),
                                                                                                result_min,
                                                                                                'dc' + str(
                                                                                                    best_position))
        resource_cpu_mem = (dicts['dc' + str(best_position)]['dc_cpu_resource'] + dicts['dc' + str(best_position)][
            'dc_mem_resource']) / 100
        migration_cost.append(distance * resource_cpu_mem)
        variance_cpu_resource, variance_mem_resource = text1.return_dcs_resource_variance()
        resource_variance = 0.5 * variance_cpu_resource + 0.5 * variance_mem_resource
        resource_variance_list.append(1 / resource_variance)
        fitness_list.append(1/best_fitness)

    text1.create_all_sfc4()
    exceed_dcs = text1.check_dc_threshold_exceed(0.5, 0.5)
    print(exceed_dcs)
    for key in exceed_dcs:
        vnf_dict = text1.GNT_VNF_choose(key)
        result_min = min(vnf_dict, key=lambda x: vnf_dict[x])
        best_position, best_fitness = dbpso(text1, key, num_particles, max_iterations, w, c1, c2)
        print("--------------------------------------------------------------------------------------------")
        print("最优解的位置：dc", best_position)
        print("最优解的适应度值：", 1/best_fitness)
        print("--------------------------------------------------------------------------------------------")
        text1.migration(best_position, key)
        distance = distance_next_vnf_in_dc(text1, text1.return_sfcs(), text1.return_dcs_link_matrix(), result_min,
                                           'dc' + str(best_position)) + distance_last_vnf_in_dc(text1,
                                                                                                text1.return_sfcs(),
                                                                                                text1.return_dcs_link_matrix(),
                                                                                                result_min,
                                                                                                'dc' + str(
                                                                                                    best_position))
        resource_cpu_mem = (dicts['dc' + str(best_position)]['dc_cpu_resource'] + dicts['dc' + str(best_position)][
            'dc_mem_resource']) / 100
        migration_cost.append(distance * resource_cpu_mem)
        variance_cpu_resource, variance_mem_resource = text1.return_dcs_resource_variance()
        resource_variance = 0.5 * variance_cpu_resource + 0.5 * variance_mem_resource
        resource_variance_list.append(1 / resource_variance)
        fitness_list.append(1/best_fitness)

    # 打印最优解和最优适应度值
    end_time = time.time()

    print("程序运行时间:", end_time - start_time)
    print(resource_variance_list)
    print(migration_cost)