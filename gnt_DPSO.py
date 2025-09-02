import numpy as np
import gnt_Util
import time

def dbpso(text1, exceed_dc, num_particles, max_iterations, w, c1, c2):
    particles_position = [0, 1, 2, 3]
    for i in range(0, len(particles_position)):
        if particles_position[i] == int(exceed_dc.split("dc")[-1]):
            if i == 3:
                particles_position[i] = particles_position[i - 1]
            if i == 0:
                particles_position[i] = particles_position[i + 1]
            else:
                particles_position[i] = particles_position[i - 1]
    particles_velocity = [0, 0, 0, 0]

    particles_best_position = [0, 0, 0, 0]
    particles_best_fitness = [100, 100, 100, 100]
    global_best_position = 0
    global_best_fitness = 100
    new_positions = [0, 0, 0, 0]
    need_migration_vnf = text1.chooose_migrate_VNF(exceed_dc)

    for iteration in range(max_iterations):
        for i in range(0, num_particles):
            fitness = binary_func(text1, particles_position[i], exceed_dc, need_migration_vnf)
            if fitness < particles_best_fitness[i]:
                particles_best_position[i] = particles_position[i]
                particles_best_fitness[i] = fitness
            if fitness < global_best_fitness:
                global_best_position = particles_position[i]
                global_best_fitness = fitness
            particles_position_binary_list = decimal_to_binary(particles_position[i], 6)
            for j in range(0, 6):
                if np.random.rand() < 0.5:  # 生成一个0到1之间的随机数，如果小于mutation_prob，则进行翻转
                    particles_position_binary_list[j] = 1 - particles_position_binary_list[j]  # 进行二进制位的翻转
            particles_position[i] = binary_to_decimal(particles_position_binary_list)  # 将二进制编码转换回十进制值
            while(particles_position[i]>49 or particles_position[i] == int(exceed_dc.split("dc")[-1])):
                for j in range(0, 6):
                    if np.random.rand() < 0.5:  # 生成一个0到1之间的随机数，如果小于mutation_prob，则进行翻转
                        particles_position_binary_list[j] = 1 - particles_position_binary_list[j]  # 进行二进制位的翻转
                particles_position[i] = binary_to_decimal(particles_position_binary_list)  # 将二进制编码转换回十进制值
    best_position = global_best_position
    best_fitness = global_best_fitness
    return "dc"+str(best_position), need_migration_vnf

# 示例：定义一个简单的二进制目标函数，输入为二进制位置，返回适应度值
def binary_func(text1, choosed_dc, exceed_dc, need_migration_vnf):
    choosed_dc = "dc" + str(choosed_dc)
    if choosed_dc!= exceed_dc:
        return text1.migration_cost(choosed_dc, exceed_dc, need_migration_vnf)
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

# 调用 DBPSO 算法
choose_dbpso_vnf = {}
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
    resource_variance_list = []
    exceed_dc_list_length = []
    delay_time_list = []
    num_particles = 4
    num_dims = 50
    max_iterations = 50
    w = 0.1
    c1 = 0.4
    c2 = 1
    migration_number = 0
    exceed_dc_list = text1.check_dc_threshold_exceed(0.5, 0.5)
    print(len(exceed_dc_list))
    print("需要迁移的dc_list：", len(exceed_dc_list))
    while len(exceed_dc_list) > 0:
        print("11111111111111111111111111111111",text1.return_dcs_dict())
        print("需要迁移的dc：", exceed_dc_list[0])
        choosed_dc, need_migration_vnf = dbpso(text1, exceed_dc_list[0], num_particles, max_iterations, w, c1, c2)
        path_len = text1.Delay_time(choosed_dc, exceed_dc_list[0], need_migration_vnf)
        print("路径长度：",path_len)
        delay_time_list.append(path_len)
        print("choosed_dc:",choosed_dc)
        text1.after_migration_dc_and_link(choosed_dc, exceed_dc_list[0], need_migration_vnf)
        print("迁移后资源利用率：",text1.Resource_Utilization(choosed_dc))
        exceed_dc_list.pop(0)
        exceed_dc_list = text1.check_dc_threshold_exceed(0.5, 0.5)
        if len(exceed_dc_list) > 0:
            if choosed_dc == exceed_dc_list[0]:
                exceed_dc_list.pop(0)
        print("列表长度为：", len(exceed_dc_list))
        exceed_dc_list_length.append(len(exceed_dc_list))
        print("22222222222222222222222222222222", text1.return_dcs_dict())
        migration_number += 1

    # 打印最优解和最优适应度值
    end_time = time.time()
    # print(resource_variance_list)
    print(migration_number)
    print("程序运行时间:", end_time - start_time)
    print(exceed_dc_list_length)
    print("delay_time_list:",delay_time_list)